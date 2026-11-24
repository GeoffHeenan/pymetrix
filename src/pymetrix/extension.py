"""The extension API (1.10): everything a plugin imports from pymetrix, besides the stable modules in ``STABLE``.

A plugin registers its calls, dialogs, results' views, help, report templates, example datasets, workspace, menu and
panels through this module alone, so pymetrix can change its internals without breaking plugins:

    from pymetrix.extension import PluginInfo, api, command, plugin_context, register, register_view

Its other imports come from the stable modules (``STABLE``): the data, tables, figures, errors, estimators, the
state-space engine, the metadata and the session's commands. ``tests`` in a plugin made by ``new_plugin`` checks that.

The app's classes are plain, with no user interface in them, so a plugin can register its workspace, menu and
panels as pymetrix itself is imported, with no window and no Qt.

    from pymetrix.plugins import register
    from pymetrix.extension import AppPanel, MenuLayout, Workspace

    register("workspaces", Workspace("finance", "Finance", ("Finance",), ("Objects", "History"), ("FitDCC",), "risk and returns"))
    register("menus", MenuLayout("Finance", ["FitDCC", None, ("Risk", ["ValueAtRisk", "Backtest VaR"])]))
    register("panels", AppPanel("Assistant", lambda app: MyWidget(app), area="right"))

A plugin that keeps its own record of the work (a lab book, an activity log) registers a ``SessionObserver``: every
session, in scripts, notebooks, the app and the MCP server, then tells it each step run, failed, cancelled, undone or
removed, and each project opened or saved (``SessionEvent``; see ``pymetrix.commands.events``). It may keep files in
the project, in ``session.attachments[key]`` (name -> bytes): they are saved and reopened with it, and kept as they
are when the plugin is missing. Each save notes which recording observers were active (``session.save_log``).

    register("observers", SessionObserver("labbook", handle))   # handle(event), for every SessionEvent

A plugin can add a line about each result a report exports (``ReportNote``): how it was reached, say. The lines go
in a section under the note's heading, before the appendix.

    register("report_notes", ReportNote("tried", "What was tried", lambda session, name: f"{name}: 1 of 23 fitted"))

A panel's ``create(app)`` receives an ``AppInterface``: the session, the selection, opening a dialog filled in,
running a step the way the app does (in the background, with Cancel and Undo, one at a time) and (1.7) opening a demo
project (``open_demo``: the app asks to save unsaved work first). Panels use only this, never the window's own parts,
so they keep working as the app changes. A panel that proposes steps (an assistant) opens each in its dialog inside
``with app.session.proposing(CommandClass, "Claude"):`` (1.9): the step the user then runs is marked
``CommandRecord.proposed_by``, saved with the project, shown in History and noted in the script. A result class tells
the app its stage
(``stage = "forecast"``) and its next steps (``next_steps = (("Label…", "CommandClass", "field", "stage"), ...)``)
as class attributes.

A plugin that translates EViews programs uses the core EViews reader's text helpers (1.10): ``eviews_blank_comments``
and ``eviews_statements`` split a program into statements with their offsets in the text; ``parse_eviews_expression``
parses one expression into a tree of ``EViewsNumber``, ``EViewsName``, ``EViewsCoefficient``, ``EViewsCall``,
``EViewsNegative``, ``EViewsBinary`` and ``EViewsBrackets`` nodes, each with its offset ``at``;
``render_eviews_expression`` writes a tree as ``convert_expression`` does; text that does not read raises
``EViewsExpressionError`` with its place; ``Diagnostic`` and ``line_and_column`` report a problem at a line and column.

    clean = eviews_blank_comments(program)
    for text, at in eviews_statements(clean):
        lhs, _, rhs = text.partition("=")
        tree = parse_eviews_expression(rhs, at + len(lhs) + 1, {"c"})   # offsets in the whole program
        python = render_eviews_expression(tree, frequency="Q")
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

# the stable modules a plugin may import from, besides this one (their public names: none starting with "_")
STABLE = ("pymetrix", "pymetrix.extension", "pymetrix.errors", "pymetrix.data", "pymetrix.output", "pymetrix.figures",
          "pymetrix.estimation", "pymetrix.statespace", "pymetrix.metadata", "pymetrix.commands")

# what plugins need from pymetrix's inside, under public names: name -> (module, attribute), loaded on first use
_PUBLIC = {
    "PluginInfo": ("pymetrix.plugins", "PluginInfo"),
    "plugin_context": ("pymetrix.plugins", "plugin_context"),
    "register": ("pymetrix.plugins", "register"),
    "api": ("pymetrix._api", "api"),  # @api("prefix:name"): a recorded, replayable call
    "make_call": ("pymetrix.calls", "make_call"),  # a dialog's call: make_call("prefix:name", output=..., **arguments)
    "Ref": ("pymetrix.codec", "Ref"),  # a workspace object by name, in a call
    "Command": ("pymetrix.commands.registry", "Command"),
    "Widget": ("pymetrix.commands.registry", "Widget"),
    "command": ("pymetrix.commands.registry", "command"),  # @command(menu="Menu/Item..."): a dialog
    "register_view": ("pymetrix.views", "register_view"),
    "figure_views": ("pymetrix.views", "_figure"),  # [("Graph", figure)] from a plot method, or [] without matplotlib
    "frame_table": ("pymetrix.views", "_frame_table"),  # a DataFrame as a Table view
    "views_frame": ("pymetrix.views", "_views_to_frame"),  # one view of a result as a DataFrame (for to_frame(view))
    "report_progress": ("pymetrix.progress", "report"),  # a long call's progress (the app's Cancel checks here too)
    "check_cancelled": ("pymetrix.progress", "check"),
    "add_help_topic": ("pymetrix.help", "add_topic"),
    "ReportTemplate": ("pymetrix.report", "ReportTemplate"),
    "numerical_hessian": ("pymetrix.estimation.numdiff", "hessian"),
    "point_estimate": ("pymetrix.estimation.params", "point"),  # a result's params: names, values and covariance
    "SessionEvent": ("pymetrix.commands.events", "SessionEvent"),  # 1.3: what happened to a session's history
    "SessionObserver": ("pymetrix.commands.events", "SessionObserver"),  # 1.3: register("observers", SessionObserver(...))
    "Actor": ("pymetrix.commands.events", "Actor"),  # 1.5: who took a step (SessionEvent.actor): a person or an agent
    "migration": ("pymetrix.commands.registry", "migration"),  # 1.4: @migration("prefix:dialog", 1), a dialog's old fields
    "call_migration": ("pymetrix._api", "call_migration"),  # 1.4: @call_migration("prefix:name", 1), a call's old arguments
    "renamed_arguments": ("pymetrix._api", "renamed_arguments"),  # 1.4: old keyword names, with a DeprecationWarning
    "random_number_generator": ("pymetrix._random", "generator"),
    "build_mcp_server": ("pymetrix.mcp_server", "build_server"),  # 1.8: pymetrix's MCP tools over a session that lives elsewhere (the app's)
    "MCPPolicy": ("pymetrix.mcp_server", "Policy"),  # 1.8: what those tools may do (code, file writes, network; the root folder)  # a numpy Generator from a seed, with the bit generator the user chose
    # 1.10: the core EViews reader's text helpers, for a plugin that translates EViews programs (offsets count from 0)
    "eviews_blank_comments": ("pymetrix.models.eviews", "_blank"),  # comments as spaces: every offset stays
    "eviews_statements": ("pymetrix.models.eviews", "_statements"),  # [(statement, offset)]; " _" joins lines
    "parse_eviews_expression": ("pymetrix.models.eviews", "_parse"),  # (text, base, vectors, comparison=False) -> a tree
    "render_eviews_expression": ("pymetrix.models.eviews", "_render"),  # a tree as convert_expression writes it
    "walk_eviews_expression": ("pymetrix.models.eviews", "_walk"),  # a tree's nodes, the root first
    "EViewsExpressionError": ("pymetrix.models.eviews", "_Refused"),  # a SpecificationError with .message, .at, .length
    "EViewsNumber": ("pymetrix.models.eviews", "_Num"),  # the tree's nodes: frozen dataclasses, each with .at
    "EViewsName": ("pymetrix.models.eviews", "_Ser"),  # a series or scalar at a lag: .name, .lag, .length
    "EViewsCoefficient": ("pymetrix.models.eviews", "_Coef"),  # c(2): .vector, .index
    "EViewsCall": ("pymetrix.models.eviews", "_Call"),  # .fn ("dlog", "@pch"), .args
    "EViewsNegative": ("pymetrix.models.eviews", "_Neg"),  # .x
    "EViewsBinary": ("pymetrix.models.eviews", "_Bin"),  # .op, .left, .right
    "EViewsBrackets": ("pymetrix.models.eviews", "_Par"),  # .x
    "Diagnostic": ("pymetrix.models.source", "Diagnostic"),  # a problem at a line and column (from 1), as check() reports
    "line_and_column": ("pymetrix.models.source", "_place"),  # (text, offset) -> (line, column), both from 1
}

__all__ = ["STABLE", "AppInterface", "AppPanel", "ExampleDataset", "MenuLayout", "ReportNote", "Workspace", *_PUBLIC]


def __getattr__(name: str) -> Any:
    if name in _PUBLIC:
        import importlib

        module, attr = _PUBLIC[name]
        value = getattr(importlib.import_module(module), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'pymetrix.extension' has no attribute {name!r}")


@dataclass(frozen=True)
class Workspace:
    """A stage of the work on the app's stage bar: its menus, panels and tool shortcuts."""

    key: str
    title: str
    menus: tuple[str, ...] | None  # the menus besides the ones always shown (None: every menu)
    docks: tuple[str, ...] | None  # the panels shown (None: as arranged)
    tools: tuple[str, ...]  # toolbar shortcuts: command class names, or "@" and an app action
    tip: str


@dataclass(frozen=True)
class ExampleDataset:
    """An example dataset a plugin ships: ``Dataset.example(name)`` reads the Parquet file at ``path``. ``path`` may be
    any path-like object; ``os.fspath`` is called on it only when the dataset is loaded, so it can download data the
    plugin may not redistribute into a cache on first use."""

    name: str
    description: str
    path: Any


@dataclass(frozen=True)
class MenuLayout:
    """The order of a plugin's own top menu: command class names, None for a separator, and (title, [entries]) for a
    submenu, as ``pymetrix.commands.menus.MENU_LAYOUT``. A plugin's dialogs in the core menus go at their end."""

    top: str
    entries: Sequence[Any] = ()
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            object.__setattr__(self, "label", self.top)


@dataclass(frozen=True)
class AppPanel:
    """A docked panel: its title, ``create(app) -> QWidget`` and where it docks first (left, right or bottom)."""

    title: str
    create: Callable[[AppInterface], Any]
    area: str = "right"
    label: str = ""

    def __post_init__(self) -> None:
        if self.area not in ("left", "right", "bottom"):
            from .errors import SpecificationError

            raise SpecificationError(f"a panel docks left, right or bottom, not {self.area!r}")
        if not self.label:
            object.__setattr__(self, "label", self.title)


@dataclass(frozen=True)
class ReportNote:
    """A line about each result a report exports (extension API 1.6): ``note(session, name)`` returns the line for the
    result called ``name``, or None for none. The lines go under ``heading``, before the appendix."""

    name: str
    heading: str
    note: Callable[[Any, str], str | None]


def _qt_owner(callback: Any) -> Any:
    """The Qt object ``callback`` is a method of (a panel's ``refresh``), or None (a plain function, or no Qt)."""
    owner = getattr(callback, "__self__", None)
    if owner is None:
        return None
    try:
        from PySide6.QtCore import QObject
    except ImportError:  # pragma: no cover - no Qt, so no panels
        return None
    return owner if isinstance(owner, QObject) else None


def _alive(obj: Any) -> bool:
    """``obj``'s C++ object is still there (False once Qt has deleted it)."""
    try:
        import shiboken6
    except ImportError:  # pragma: no cover
        return True
    return shiboken6.isValid(obj)


@dataclass
class AppInterface:
    """The desktop app as a plugin's panel sees it."""

    _session: Callable[[], Any]
    _selected: Callable[[], str | None]
    _open_dialog: Callable[[type, Any], Any]
    _run: Callable[[Any], Any]
    _status: Callable[[str], None]
    _listeners: list[Callable[[], None]] = field(default_factory=list)
    _open_demo: Callable[[str], None] | None = None  # 1.7; None: this app cannot open demos (open_demo says so)
    _closed: bool = False  # its window has closed (``_close``)

    @property
    def session(self) -> Any:
        """The open session (it changes when another project opens: read it each time)."""
        return self._session()

    def selected(self) -> str | None:
        """The name of the object selected in the Objects panel, if any."""
        return self._selected()

    def open_dialog(self, command_cls: type, initial: Any = None) -> Any:
        """Open a command's dialog, filled in from ``initial`` (a command); the user runs it, or not."""
        return self._open_dialog(command_cls, initial)

    def run(self, command: Any) -> Any:
        """Run a step as the app does: recorded, in the background with progress and Cancel, undoable, and refused
        while another step runs. Returns its result; raises its error."""
        return self._run(command)

    def status(self, message: str) -> None:
        """A message in the status bar."""
        self._status(message)

    def open_demo(self, name: str) -> None:
        """Open the demo project ``name`` (``pymetrix.demos()``) as it first was, as File → Open demo project does:
        the app asks to save unsaved work first, and nothing changes if the user cancels (extension API 1.7).

        Raises ``NotFoundError`` (with the demos there are) for an unknown name, and ``NotSupportedError`` from an
        interface made without a way to open demos; a panel written for 1.5 too checks for the method, or catches it."""
        if self._open_demo is None:
            from .errors import NotSupportedError

            raise NotSupportedError("this app cannot open demo projects from a panel: use File → Open demo project…")
        self._open_demo(name)

    def on_history_changed(self, callback: Callable[[], None]) -> None:
        """Call ``callback`` whenever a step is run, undone or the session replaced. A method of a Qt object (a
        panel's ``refresh``) is called while that object lives: once it is deleted it is dropped, not called."""
        if self._closed:
            return  # its window has closed: nothing of it follows the history any more
        self._listeners.append(callback)
        owner = _qt_owner(callback)
        if owner is not None:
            owner.destroyed.connect(lambda *_: self._drop(callback))

    def _drop(self, callback: Callable[[], None]) -> None:
        self._listeners[:] = [c for c in self._listeners if c is not callback and c != callback]

    def _close(self) -> None:
        """The app's window has closed: its panels stop following the history (a later window's code, run in the
        console all windows share, must not refresh a closed window's panels, whose widgets Qt may have deleted)."""
        self._closed = True
        self._listeners.clear()

    def _changed(self) -> None:
        if self._closed:
            return
        for callback in list(self._listeners):
            owner = _qt_owner(callback)
            if owner is not None and not _alive(owner):
                self._drop(callback)  # its panel is gone (deleted without the app seeing it): never touch its widgets
                continue
            try:
                callback()
            except Exception:  # noqa: BLE001 - a panel's mistake must not stop the app refreshing
                pass
