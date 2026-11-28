"""PySide6 desktop frontend: menus and dialogs generated from the command registry.

Run with ``pymetrix-gui`` after ``pip install pymetrix[qt]``.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

from pymetrix_app import NAME

from ..commands import COMMANDS, Command, FieldSpec, Session, form_spec
from .keys import key_text
from .messages import EMPTY_STATES, plural, titled
from . import metrics, theme
from .icons import icon as kind_icon, object_icon, size_icons  # each object's kind, at a glance


FIELDS_VIEW = "Python access"  # a result's last view: its fields and the Python that reaches each (was "Fields")


def _qt():
    try:
        from PySide6 import QtWidgets
    except ImportError as exc:
        raise ImportError("the desktop UI needs PySide6: pip install pymetrix[qt]") from exc
    return QtWidgets


def default_dock_size(area: str, height: int, width: int) -> int:
    """The size a panel first takes in a main window of ``width`` by ``height`` (its default layout, before the user
    arranges it): the bottom panels (Script, Immediate, Problems) about a fifth of the height, at most 240 pixels and
    at least 120, so the result windows keep most of a laptop screen (1366 by 768, 1280 by 720); the left panels
    (Objects, History) a fifth of the width, 260 to 360 pixels, wide enough for an object's name and type."""
    if area == "bottom":
        return max(120, min(240, round(0.2 * height)))
    return max(260, min(360, round(0.2 * width)))


def _dock_class(Qt: Any) -> Any:
    """QDockWidget whose size hint is ``default_dock_size`` in the bottom and left areas: what a main window gives a
    panel it places for the first time (a first run, a workspace never arranged, Reset layout). A layout saved by
    the user (restoreState) keeps its own sizes."""
    cached = _DOCK_CLASS.get("cls")
    if cached is not None:
        return cached
    from PySide6.QtCore import QSize
    from PySide6.QtCore import Qt as QtNS

    class Dock(Qt.QDockWidget):
        def sizeHint(self) -> Any:  # noqa: N802
            hint = super().sizeHint()
            main = self.parentWidget()
            if not isinstance(main, Qt.QMainWindow) or self.isFloating():
                return hint
            area = main.dockWidgetArea(self)
            screen = (main.screen() or self.screen()).availableGeometry()
            height = main.height() if main.isVisible() else screen.height()
            width = main.width() if main.isVisible() else screen.width()
            if area == QtNS.DockWidgetArea.BottomDockWidgetArea:
                return QSize(hint.width(), default_dock_size("bottom", height, width))
            if area == QtNS.DockWidgetArea.LeftDockWidgetArea:
                return QSize(default_dock_size("left", height, width), hint.height())
            return hint

    _DOCK_CLASS["cls"] = Dock
    return Dock


_DOCK_CLASS: dict[str, Any] = {}


def _scope_note(session: Any, dataset: str) -> str | None:
    """"24 of 2,140 series (working set: gdp_*)" for a dataset with a working set; None without one."""
    shown = session.working_set(dataset) if hasattr(session, "working_set") else None
    if shown is None:
        return None
    total = len(session[dataset].names)
    return f"{len(shown):,} of {total:,} series (working set: {session.working_sets[dataset]})"


_MENUS: list[Any] = []  # every menu the window made, kept: PySide6 6.6 deleted menus made by addMenu(title)


def owned_menu(parent: Any, title: str) -> Any:
    """A menu titled ``title`` under ``parent`` (a menu bar or a menu), owned by it and kept alive."""
    menu = _qt().QMenu(title, parent)
    parent.addMenu(menu)
    _MENUS.append(menu)
    return menu


# The menu bar's access keys (Alt+F opens File...): each letter used once; a plugin's menu takes the first free letter
ACCESS_KEYS = {"File": "&File", "Edit": "&Edit", "Data": "&Data", "Estimate": "E&stimate", "Test": "&Test", "Model": "&Model",
               "Forecast": "F&orecast", "View": "&View", "Calculate": "&Calculate", "Help": "&Help"}


def with_access_key(title: str, taken: str = "") -> str:
    """A top menu's title with its access key marked ("&File"); a plugin's menu (not in ``ACCESS_KEYS``) takes the
    first of its letters that no core menu (nor ``taken``) uses, or none."""
    if title in ACCESS_KEYS:
        return ACCESS_KEYS[title]
    used = {t[t.index("&") + 1].lower() for t in ACCESS_KEYS.values()} | set(taken.lower())
    for i, letter in enumerate(title):
        if letter.isalpha() and letter.lower() not in used:
            return title[:i] + "&" + title[i:]
    return title


def mark_access_keys(menu: Any) -> None:
    """Give every line of ``menu`` and of its submenus an access key, unique within its menu ("&Kalman filter…"): the
    lines that mark their own keep theirs; see pymetrix.commands.menus.with_access_keys."""
    from ..commands.menus import with_access_keys

    actions = [a for a in menu.actions() if not a.isSeparator() and a.text()]
    # a line keyed here before is keyed afresh from its own text, so a line added since (a window opened) can have a
    # letter an earlier line gives up; a line whose text changed since (Undo's step) starts from its new text
    plain = [a.property("pymetrix_plain") if a.property("pymetrix_keyed") == a.text() else a.text() for a in actions]
    for action, before, text in zip(actions, plain, with_access_keys(plain)):
        if text != action.text():
            action.setText(text)
        action.setProperty("pymetrix_plain", before)
        action.setProperty("pymetrix_keyed", text)
    for action in actions:
        if action.menu() is not None and action.menu() is not menu:
            keep_access_keys(action.menu())


def keep_access_keys(menu: Any) -> None:
    """``mark_access_keys`` now and each time ``menu`` opens, for the lines a menu adds or renames as it opens (Open
    recent, the open windows, Undo's step). Connected last, so it runs after the menu's own filling."""
    if not menu.property("pymetrix_access_keys"):
        menu.setProperty("pymetrix_access_keys", True)
        menu.aboutToShow.connect(lambda m=menu: mark_access_keys(m))
    mark_access_keys(menu)


_PROGRESS: dict[str, Any] = {}  # the main window's progress bar: "show", "update" and "hide" (set by main_window), and "window"

# show() in the app: the newest open window shows results. Each window is a token here (not the window: a reference
# would keep it alive), oldest first, with its handler; closing a window takes out its own entry only, so windows closed
# in any order leave show() with the newest window still open, then, with none open, as it was before the first opened
# (printing). Restoring what each window found when it opened left a closed window's handler in place when an older
# window closed first, so show() went on to a window that no longer existed and printed nothing.
_SHOWING: list[tuple[object, Any]] = []
_SHOW_BEFORE: list[Any] = []  # show()'s handler before the first of the open windows opened


def _show_in(token: object, handler: Any) -> None:
    """Route show() to a window that has opened (``token`` stands for it)."""
    from ..output import set_show_handler

    previous = set_show_handler(handler)
    if not _SHOWING:
        _SHOW_BEFORE[:] = [previous]
    _SHOWING.append((token, handler))


def _show_closed(token: object) -> None:
    """A window has closed (or been destroyed): show() goes to the newest window still open, else back to what it did
    before the first opened. A second call for the same window does nothing."""
    from ..output import set_show_handler

    if not any(t is token for t, _ in _SHOWING):
        return
    _SHOWING[:] = [(t, h) for t, h in _SHOWING if t is not token]
    set_show_handler(_SHOWING[-1][1] if _SHOWING else (_SHOW_BEFORE[0] if _SHOW_BEFORE else None))


def _belongs_to(widget: Any, window: Any) -> bool:
    """Whether ``widget`` is part of ``window`` (False for a widget already deleted)."""
    try:
        while widget is not None:
            if widget is window:
                return True
            widget = widget.parentWidget()
    except RuntimeError:  # deleted: it belongs to nothing now
        return False
    return False
_BUSY: dict[str, Any] = {"running": False}  # a step is running in the background: no other may start


def busy() -> bool:
    """Whether a step is running in the background."""
    return _BUSY["running"]


def cancel_running() -> None:
    """Ask the step running in the background to stop (at its next check)."""
    if "cancel" in _PROGRESS:
        try:
            _PROGRESS["cancel"]()  # the status bar says so
        except RuntimeError:
            _PROGRESS.clear()
    if "task" in _BUSY:
        _BUSY["task"].cancel()


_GC_PAUSE: dict[str, Any] = {"depth": 0, "resume": False}  # steps running (nested ones too); collect when they end


def _pause_collection() -> None:
    """Pause automatic garbage collection and freeze what exists now (the first of nested steps only)."""
    import gc

    if _GC_PAUSE["depth"] == 0:
        _GC_PAUSE["resume"] = gc.isenabled()
        gc.disable()
        gc.freeze()  # moved out of every collection's reach (explicit ones too) until _resume_collection
    _GC_PAUSE["depth"] += 1


def _resume_collection() -> None:
    """Undo ``_pause_collection`` once the last running step has ended: unfreeze, and resume automatic collection."""
    import gc

    _GC_PAUSE["depth"] = max(0, _GC_PAUSE["depth"] - 1)
    if _GC_PAUSE["depth"] == 0:
        gc.unfreeze()
        from . import collector

        if _GC_PAUSE["resume"] and not collector.active():  # with the GUI thread collecting, it stays off
            gc.enable()


def run_blocking(fn: Any, label: str = "") -> Any:
    """Run ``fn`` (a step) on a worker thread while the window stays live, and return its result (or raise its
    error) when it is done. After a moment the status bar shows its progress and Cancel (``pymetrix.progress``).

    Python's cyclic garbage collector runs on whichever thread happens to allocate when it is due. Run on the worker
    thread, it can finalize Qt objects left over from earlier windows and dialogs there, off the GUI thread: a crash
    (a segmentation fault) or a broken event loop (timers no longer firing). So automatic collection is paused while
    the worker runs, young garbage is collected on the GUI thread as the window waits, and the worker has finished
    (joined) before collection resumes.

    Pausing alone is not enough: a collection can still start off the GUI thread during the step. JPype runs a full
    collection on a Java thread whenever Java's collector has run (once TRAMO-SEATS has started the JVM), and a library
    can turn automatic collection back on meanwhile (parso's cache does so unconditionally; an import's disable/enable
    bracket on another thread can span the step's start). So the objects that exist when the step starts are also
    frozen (``gc.freeze``) until it ends: no collection, on any thread, looks at them, and the Qt objects of earlier
    windows are left for the GUI thread. The GUI thread also turns automatic collection off again if it finds it on.
    See ``_pause_collection``."""
    import gc
    import threading
    import time

    from PySide6.QtCore import QEventLoop, QTimer

    from ..progress import task

    result: dict[str, Any] = {}
    done = threading.Event()
    worker: threading.Thread | None = None
    _BUSY["running"] = True
    _pause_collection()
    try:
        with task() as current_task:
            _BUSY["task"] = current_task

            def work() -> None:
                try:
                    result["value"] = fn()
                except BaseException as exc:  # noqa: BLE001 - raised again where the step was asked for
                    result["error"] = exc
                finally:
                    done.set()

            worker = threading.Thread(target=work, name="pymetrix step", daemon=True)
            worker.start()
            loop = QEventLoop()
            started = time.perf_counter()
            shown = [False]

            def tick() -> None:
                if done.is_set():
                    loop.quit()
                    return
                if gc.isenabled():  # turned on again meanwhile (a library): off until the worker has finished
                    _GC_PAUSE["resume"] = True
                    gc.disable()
                if _GC_PAUSE["resume"]:  # the collections the paused collector would have made, here on the GUI thread
                    counts, thresholds = gc.get_count(), gc.get_threshold()
                    if thresholds[0] and counts[0] > thresholds[0]:
                        gc.collect(1 if counts[1] >= thresholds[1] else 0)
                try:
                    if not shown[0] and time.perf_counter() - started > 0.4 and "show" in _PROGRESS:
                        _PROGRESS["show"](current_task, label)
                        shown[0] = True
                    elif shown[0] and "update" in _PROGRESS:
                        _PROGRESS["update"](current_task)
                except RuntimeError:  # the window that showed progress is gone
                    _PROGRESS.clear()

            timer = QTimer()
            timer.setInterval(100)
            timer.timeout.connect(tick)
            timer.start()
            if not done.wait(0.05):  # a quick step returns at once; a long one keeps the window live meanwhile
                loop.exec()
            timer.stop()
            if shown[0] and "hide" in _PROGRESS:
                try:
                    _PROGRESS["hide"]()
                except RuntimeError:
                    _PROGRESS.clear()
    finally:
        if worker is not None and done.is_set():
            worker.join()  # it has set its result: only its own clean-up is left, which ends now, before collecting resumes
        _resume_collection()
        _BUSY["running"] = False
        finished = _BUSY.pop("task", None)
        _BUSY["cancel ignored"] = finished is not None and finished.cancelled and "error" not in result
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _form(Qt: Any, parent: Any = None) -> Any:
    """A dialog's form layout: each field beside its label where there is room, under it where the two side by side
    would be wider than the form (WrapLongRows), so the form's least width is its widest label or field, not the widest
    label and widest field together. Side by side only, Reconcile's form (the label "Replace the components in the
    data" and the method's radio choice "Least squares (in proportion to the values)") and Fit VECM's (the Advanced
    section's "Exogenous series (unrestricted)" and the deterministic terms' longest choice) were wider than a
    1366-pixel screen where every character is a 16-pixel box (Windows CI with the font a third bigger), and scrolled
    sideways."""
    form = Qt.QFormLayout() if parent is None else Qt.QFormLayout(parent)
    form.setRowWrapPolicy(Qt.QFormLayout.RowWrapPolicy.WrapLongRows)
    return form


class _Editor:
    """One form field: its widget, how to read it, refill its choices, and watch for changes."""

    def __init__(self, Qt: Any, spec: FieldSpec):
        self.Qt, self.spec = Qt, spec
        default = spec.default
        if spec.kind == "options":
            self.widget = Qt.QWidget()
            self.inner = _form(Qt, self.widget)
            metrics.set_margins(self.inner, "indent", "none", "none", "none")  # under the choice that opens it
            self.sub: dict[str, _Editor] = {}
            self.sub_specs: list[FieldSpec] = []
            self.for_value: Any = object()
            self.advanced_part = Qt.QWidget()  # the chosen extension's advanced settings: in the dialog's Advanced section
            self.advanced_form = _form(Qt, self.advanced_part)
            metrics.set_margins(self.advanced_form, "indent", "none", "none", "none")
        elif spec.multiple:
            self.widget = Qt.QListWidget()
            self.widget.setMaximumHeight(120)
            self.picked = set(s.strip() for s in str(default or "").split(",") if s.strip())
        elif spec.dynamic or (spec.kind == "choice" and not spec.radio):
            self.widget = Qt.QComboBox()
            if not spec.dynamic:
                for c in spec.choices or ():  # a choice with no fixed options: an empty list, not a crash
                    self.widget.addItem(spec.display(c), c)
                self.widget.setCurrentIndex(max(0, self.widget.findData(default)))
        elif spec.radio:
            from .fit import FlowRow

            self.widget = FlowRow()  # its choices on one line where there is room, on more where there is not
            self.group = Qt.QButtonGroup(self.widget)
            for choice in spec.choices:
                button = Qt.QRadioButton(spec.display(choice))
                button.choice = choice
                button.setChecked(choice == default)
                self.group.addButton(button)
                self.widget.addWidget(button)
        elif spec.kind == "checkbox":
            self.widget = Qt.QCheckBox()
            self.widget.setChecked(bool(default))
        elif spec.kind in ("file", "savefile"):
            self.widget = Qt.QWidget()
            row = Qt.QHBoxLayout(self.widget)
            row.setContentsMargins(0, 0, 0, 0)
            self.path = Qt.QLineEdit("" if default is None else str(default))
            self.path.setPlaceholderText(spec.help or "choose a file")
            browse = Qt.QPushButton("Browse…")

            def pick() -> None:
                ask = Qt.QFileDialog.getSaveFileName if spec.kind == "savefile" else Qt.QFileDialog.getOpenFileName
                name, _ = ask(self.widget, "Save as" if spec.kind == "savefile" else "Choose a file", self.path.text(), spec.source or "All files (*)")
                if name:
                    self.path.setText(name)

            browse.clicked.connect(pick)
            row.addWidget(self.path, 1)
            row.addWidget(browse)
            self.browse = browse
        elif spec.kind == "number" and spec.scientific:
            self.widget = Qt.QLineEdit(repr(default) if default is not None else "")
            from PySide6.QtGui import QDoubleValidator

            self.widget.setValidator(QDoubleValidator())
        elif spec.kind == "number":
            number, self.optional = _number_type(spec.type)
            low, high, step = spec.bounds()
            if number is int and spec.min is None and spec.type is not int:
                low = 0  # an optional whole number: from 0 unless its widget says
            self.widget = Qt.QSpinBox() if number is int else Qt.QDoubleSpinBox()
            if number is int:
                low, high, step = int(low), int(high), int(step) or 1
            else:
                self.widget.setDecimals(4)
            if self.optional:  # one step below the range reads as what leaving it empty means, and gives None
                low = low - step
                self.widget.setSpecialValueText(_empty_means(spec.help) or "automatic")
            self.widget.setRange(low, high)
            self.widget.setSingleStep(step)
            zero = _zero_means(spec.help)
            if zero and low <= 0 <= high and not self.optional:
                self.widget.setSpecialValueText(zero)  # "automatic", not a bare 0
            self.widget.setValue(low if self.optional and default is None else (default or 0))
        elif spec.kind == "period":
            role = "start" if spec.name.endswith("start") else "end" if spec.name.endswith("end") else None
            self.widget = _period_spin(Qt)("" if default is None else str(default), role)
            from ..forms import needs_value

            self.widget.required = needs_value(spec)
        elif spec.kind == "multiline":
            self.widget = Qt.QPlainTextEdit("" if default is None else str(default))
            self.widget.setPlaceholderText(spec.help)
            self.widget.setMinimumHeight(80)
        else:
            self.widget = Qt.QLineEdit("" if default is None else str(default))
            self.widget.setPlaceholderText(short_placeholder(spec.help))  # the whole help in the tooltip and F1
        if spec.help:
            self.widget.setToolTip(spec.help)
        _choice_tooltips(self)
        self.outer = self.widget  # what the form shows: the widget, or the widget with its search and scope rows
        self.show_all = False  # past the dataset's working set
        self.on_scope: Any = None
        if spec.kind == "series":
            self._series_tools(Qt)
        if spec.kind == "formula" or (spec.kind == "text" and spec.name in ("formula", "where")):
            self._message_line(Qt)

    def _message_line(self, Qt: Any) -> None:
        """A formula's line under it, for what is wrong with it as typed (pymetrix.ui.formula_check); hidden while
        there is nothing to say."""
        from PySide6.QtCore import Qt as QtNS

        outer = Qt.QWidget()
        box = Qt.QVBoxLayout(outer)
        box.setContentsMargins(0, 0, 0, 0)
        metrics.set_spacing(box, "hairline")  # the field and its own message line, together
        box.addWidget(self.widget)
        self.message = Qt.QLabel()
        self.message.setWordWrap(True)
        self.message.setTextInteractionFlags(QtNS.TextInteractionFlag.TextSelectableByMouse)
        self.message.setVisible(False)
        box.addWidget(self.message)
        self.outer = outer

    def _series_tools(self, Qt: Any) -> None:
        """A series list made for large datasets: a filter (and, for ticks, Ticked only and a count) above it, and
        under it, when the dataset has a working set, how many series it shows, with Show all."""
        from PySide6.QtCore import Qt as QtNS

        outer = Qt.QWidget()
        box = Qt.QVBoxLayout(outer)
        box.setContentsMargins(0, 0, 0, 0)
        metrics.set_spacing(box, "hairline")  # the list, its filter and its scope line, together
        if self.spec.multiple:
            tools = Qt.QHBoxLayout()
            self.search = Qt.QLineEdit()
            self.search.setPlaceholderText("Filter series…")
            self.search.setClearButtonEnabled(True)
            self.ticked_only = Qt.QCheckBox("Ticked only")
            self.count = Qt.QLabel()
            self.count.setStyleSheet(theme.muted_css())
            tools.addWidget(self.search, 1)
            tools.addWidget(self.ticked_only)
            tools.addWidget(self.count)
            box.addLayout(tools)
            self.widget.setMaximumHeight(160)
            self.search.textChanged.connect(lambda *_: self._apply_filter())
            self.ticked_only.toggled.connect(lambda *_: self._apply_filter())
            self.on_tick(self._apply_filter)
        else:  # a dropdown that narrows as you type
            self.widget.setEditable(True)
            self.widget.setInsertPolicy(Qt.QComboBox.InsertPolicy.NoInsert)
            completer = self.widget.completer()
            completer.setFilterMode(QtNS.MatchFlag.MatchContains)
            completer.setCompletionMode(Qt.QCompleter.CompletionMode.PopupCompletion)
            completer.setCaseSensitivity(QtNS.CaseSensitivity.CaseInsensitive)
        box.addWidget(self.widget)
        scope = Qt.QHBoxLayout()
        self.scope_note = Qt.QLabel()
        self.scope_note.setStyleSheet(theme.muted_css())
        self.scope_button = Qt.QPushButton("Show all")
        self.scope_button.setFlat(True)
        self.scope_button.setStyleSheet(theme.css("link") + " text-decoration: underline;")
        self.scope_button.clicked.connect(self._toggle_scope)
        scope.addWidget(self.scope_note)
        scope.addWidget(self.scope_button)
        scope.addStretch()
        self.scope_row = Qt.QWidget()
        self.scope_row.setLayout(scope)
        scope.setContentsMargins(0, 0, 0, 0)
        self.scope_row.setVisible(False)
        box.addWidget(self.scope_row)
        self.outer = outer

    def _toggle_scope(self) -> None:
        self.show_all = not self.show_all
        if self.on_scope is not None:
            self.on_scope()

    def set_scope(self, note: str | None) -> None:
        """Say how many series the working set shows (None: no working set)."""
        if not hasattr(self, "scope_row"):
            return
        self.scope_row.setVisible(note is not None)
        if note is not None:
            self.scope_note.setText(note)
            self.scope_button.setText("Working set only" if self.show_all else "Show all")

    def on_tick(self, handler: Any) -> None:
        """Run ``handler()`` when a tick in this list changes. The list has one Python slot for its itemChanged, which
        runs every handler in turn (``ticked``). A handler may refill the list (the dialog's refresh does, through
        ``set_options``), deleting its items; PySide converts the changed item for each slot it calls, so a second slot
        after that one was handed a deleted item, a use-after-free that crashed with PySide6 6.6 and on macOS (Estimate
        > Cointegrating regression, whose row condition added the formula check's slot after the refresh)."""
        handlers = self.__dict__.get("tick_handlers")
        if handlers is None:
            handlers = self.tick_handlers = []
            self.widget.itemChanged.connect(lambda *_: self.ticked())
        handlers.append(handler)

    def ticked(self) -> None:
        """Each tick handler in turn, in the order they were added (``on_tick``)."""
        for handler in list(self.__dict__.get("tick_handlers", ())):
            handler()

    def _apply_filter(self) -> None:
        """Show the series that match the filter (and are ticked, with Ticked only); count the ticks."""
        if not self.spec.multiple or not hasattr(self, "search"):
            return
        words = self.search.text().lower().split()
        ticked = self.Qt_checked()
        w = self.widget
        shown = chosen = 0
        for i in range(w.count()):
            item = w.item(i)
            on = item.checkState() == ticked
            hide = (bool(words) and not all(word in item.text().lower() for word in words)) or (self.ticked_only.isChecked() and not on)
            item.setHidden(hide)
            shown += not hide
            chosen += on
        self.count.setText(f"{chosen} ticked of {w.count()}" if not words else f"{shown} shown, {chosen} ticked of {w.count()}")

    def get(self) -> Any:
        w, spec = self.widget, self.spec
        if spec.kind == "options":
            from ..commands import option_string

            return option_string(self.sub_specs, {k: e.get() for k, e in self.sub.items()})
        if spec.multiple:
            items = [w.item(i) for i in range(w.count())]
            return ", ".join(i.text() for i in items if i.checkState() == self.Qt_checked())
        if isinstance(w, self.Qt.QComboBox):
            if not spec.dynamic and w.currentIndex() >= 0:
                return w.currentData()
            text = w.currentText()
            return self._typed("" if text == self._blank() else text)
        if spec.radio:
            button = self.group.checkedButton()
            return button.choice if button else ""
        if spec.kind == "checkbox":
            return w.isChecked()
        if spec.kind in ("file", "savefile"):
            return self.path.text().strip()
        if spec.kind == "number" and spec.scientific:
            text = w.text().strip()
            try:
                return float(text) if text else spec.default
            except ValueError:  # typed half way (1e-): the text, which the form reports as not valid
                return text
        if spec.kind == "number":
            if getattr(self, "optional", False) and w.value() == w.minimum():
                return None
            value = w.value()
            return int(value) if isinstance(w, self.Qt.QSpinBox) else value
        if spec.kind == "multiline":
            return w.toPlainText()
        if spec.kind == "period":
            return w.lineEdit().text().strip()
        return w.text()

    def _blank(self) -> str:
        """How an empty choice reads in a list: what leaving it empty means, or (none)."""
        return f"({self.spec.empty})" if self.spec.empty else "(none)"

    def _typed(self, text: str) -> Any:
        """The choice itself (e.g. 0.05), not its label, for fixed choice lists."""
        for choice in self.spec.choices or ():
            if str(choice) == text:
                return choice
        return text

    def Qt_checked(self) -> Any:
        from PySide6.QtCore import Qt as QtNS

        return QtNS.CheckState.Checked

    def set(self, value: Any) -> None:
        """Show ``value`` (reopening a step's dialog as it was). Choice lists filled later keep it as their pick."""
        w, spec = self.widget, self.spec
        if spec.kind == "options":
            self.pending = str(value or "")
            self.for_value = object()  # rebuild, and apply the settings
        elif spec.multiple:
            from PySide6.QtCore import Qt as QtNS

            self.picked = {s.strip() for s in str(value or "").split(",") if s.strip()}
            w.blockSignals(True)  # one change for the whole set, not a refresh after each tick
            for i in range(w.count()):
                item = w.item(i)
                item.setCheckState(QtNS.CheckState.Checked if item.text() in self.picked else QtNS.CheckState.Unchecked)
            w.blockSignals(False)
            if w.count():
                self.ticked()  # the handlers once for the whole set, with no item handed on (see on_tick)
        elif isinstance(w, self.Qt.QComboBox):
            self.wanted = str(value)
            index = w.findData(value) if not spec.dynamic else -1
            shown = self._blank() if spec.dynamic and value in ("", None) else str(value)
            w.setCurrentIndex(index) if index >= 0 else w.setCurrentText(shown)
        elif spec.radio:
            for button in self.group.buttons():
                button.setChecked(button.choice == value or str(button.choice) == str(value))
        elif spec.kind == "checkbox":
            w.setChecked(bool(value))
        elif spec.kind in ("file", "savefile"):
            self.path.setText(str(value or ""))
        elif spec.kind == "number" and spec.scientific:
            w.setText(repr(value) if value is not None else "")
        elif spec.kind == "number":
            w.setValue(w.minimum() if getattr(self, "optional", False) and value is None else (value or 0))
        elif spec.kind == "period":
            w.lineEdit().setText("" if value is None else str(value))
        else:
            w.setPlainText("" if value is None else str(value)) if spec.kind == "multiline" else w.setText("" if value is None else str(value))

    def set_options(self, options: list[str]) -> None:
        w = self.widget
        w.blockSignals(True)
        if self.spec.multiple:
            from PySide6.QtCore import Qt as QtNS

            if w.count():
                self.picked = set(s.strip() for s in self.get().split(",") if s.strip())
            names = [*options, *sorted(p for p in self.picked if p not in options and self.spec.kind == "series")]
            if [w.item(i).text() for i in range(w.count())] == names:  # the same list: its items kept, ticks as picked
                for i in range(w.count()):
                    item = w.item(i)
                    item.setCheckState(QtNS.CheckState.Checked if item.text() in self.picked else QtNS.CheckState.Unchecked)
            else:
                w.clear()
                for name in names:
                    item = self.Qt.QListWidgetItem(name)
                    item.setFlags(item.flags() | QtNS.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(QtNS.CheckState.Checked if name in self.picked else QtNS.CheckState.Unchecked)
                    w.addItem(item)
        else:
            current = getattr(self, "wanted", None) or w.currentText() or str(self.spec.default or "")
            if self.spec.kind == "series" and current and current not in options and current != self._blank():
                options = [*options, current]  # the series chosen stays chosen, outside the working set too
            self.wanted = None
            blank = self._blank()
            shown = [blank if o == "" else o for o in options]
            w.clear()
            w.addItems(shown)
            if current == "" and blank in shown:
                current = blank
            if current in shown:
                w.setCurrentText(current)
            if not options or options == [""]:  # in words, not the kinds' class names (the line under it says more)
                w.setPlaceholderText(f"({self.spec.empty})" if self.spec.empty else "(none yet)" if self.spec.kind == "object"
                                     else "(none available)")
        w.blockSignals(False)
        self._apply_filter()

    def rebuild(self, specs: list[FieldSpec], key: Any) -> None:
        """Options sub-form: one editor per setting of the chosen test; its advanced settings go to
        ``advanced_part``, which the dialog shows in its one Advanced settings section."""
        Qt = self.Qt
        if key == self.for_value:
            return
        self.for_value = key
        while self.inner.rowCount():
            self.inner.removeRow(0)
        while self.advanced_form.rowCount():
            self.advanced_form.removeRow(0)
        self.sub_specs, self.sub = specs, {}
        from ..forms import needs_value

        for spec in specs:
            editor = _Editor(Qt, spec)
            self.sub[spec.name] = editor
            form = self.advanced_form if spec.advanced else self.inner
            form.addRow(spec.label + (" *" if needs_value(spec) else ""), editor.outer)
            _python_hint_on_label(form, editor)
        if not specs:
            self.inner.addRow(Qt.QLabel("(no settings)"))
        pending, self.pending = getattr(self, "pending", ""), ""
        if pending:  # settings of a reopened step
            from ..forms import parse_options

            _, values = parse_options(pending)
            for name, value in values.items():
                if name in self.sub:
                    self.sub[name].set(value)

    def _qtns(self) -> Any:
        from PySide6.QtCore import Qt as QtNS

        return QtNS

    def connect(self, callback: Any) -> None:
        w, spec = self.widget, self.spec
        if spec.kind == "options":
            return
        if spec.kind in ("file", "savefile"):
            self.path.textChanged.connect(lambda *_: callback())
            return
        if spec.multiple:
            self.on_tick(callback)
        elif isinstance(w, self.Qt.QComboBox) and w.isEditable():
            w.currentIndexChanged.connect(lambda *_: callback())  # a series chosen, not each key typed
        elif isinstance(w, self.Qt.QComboBox):
            w.currentTextChanged.connect(lambda *_: callback())
        elif spec.radio:
            self.group.buttonToggled.connect(lambda *_: callback())
        elif spec.kind == "checkbox":
            w.toggled.connect(lambda *_: callback())
        elif spec.kind == "number" and not spec.scientific:
            w.valueChanged.connect(lambda *_: callback())
        elif spec.kind == "period":
            w.lineEdit().textChanged.connect(lambda *_: callback())
        else:
            w.textChanged.connect(lambda *_: callback())


# ----------------------------------------------------------------- presentation: tooltips and the Advanced section

def _advanced_presentation(dialog: Any, advanced: Any, specs: list[FieldSpec], editors: dict[str, Any], initial: Any,
                           refresh: Any, shown: dict[str, bool] | None = None) -> None:
    """The dialog's Advanced settings section, for tests and scripting (``pymetrix.ui.advanced_section.present``):
    ``advanced_toggle``, ``show_advanced(on)``, ``in_advanced(name)`` (``on_advanced_tab``, its older name). Reopening
    a step whose advanced settings differ from their defaults opens the section, so what the step did is in view."""
    from .advanced_section import present

    present(dialog, advanced, specs, editors, initial, refresh, shown if shown is not None else {})


def _choice_tooltips(editor: _Editor) -> None:
    """The field's tooltip ends with how scripts write it ("Python: exog="), and each fixed choice's tooltip gives the
    code a script records ("Python: method='fmols'"), since the form shows the choice's name."""
    from PySide6.QtCore import Qt as QtNS

    from ..wording import python_hint

    spec, w = editor.spec, editor.widget
    hint = python_hint(spec.name)
    if spec.kind != "options":
        w.setToolTip(f"{spec.help}\n{hint}" if spec.help else hint)
    if not spec.choices or spec.dynamic:
        return
    if spec.radio:
        for button in editor.group.buttons():
            button.setToolTip(f"{hint}{button.choice!r}")
    elif isinstance(w, editor.Qt.QComboBox):
        for i in range(w.count()):
            w.setItemData(i, f"{hint}{w.itemData(i)!r}", QtNS.ItemDataRole.ToolTipRole)


def _python_hint_on_label(form: Any, editor: _Editor) -> None:
    """The row's label carries the field's tooltip too: hovering "Exogenous series" says "Python: exog="."""
    from ..wording import python_hint

    label = form.labelForField(editor.outer)
    if label is not None:
        label.setToolTip(editor.widget.toolTip() if editor.spec.kind != "options" else python_hint(editor.spec.name))


def _has_main_rows(editor: _Editor) -> bool:
    """A sub-form whose settings are all advanced has nothing on the form itself: its row is left out (the settings
    show in the Advanced settings section)."""
    return editor.spec.kind != "options" or editor.inner.rowCount() > 0


from .advanced_section import AdvancedSection as _AdvancedSection  # noqa: E402  (the one collapsed section: its own module)


_SPIN: dict[str, Any] = {}


def _period_spin(Qt: Any) -> Any:
    """A spin box over a dataset's periods: arrows step through them, typing still works, empty means no limit."""
    if "cls" in _SPIN:
        return _SPIN["cls"]

    class PeriodSpin(Qt.QAbstractSpinBox):
        """``role`` "start" or "end": empty stands for the first or the last period (shown greyed), and the arrows
        start from there; other period fields start from whichever end the arrow points to."""

        def __init__(self, text: str = "", role: str | None = None) -> None:
            super().__init__()
            self.periods: list[str] = []
            self.bounds: tuple[str, str] | None = None
            self.role = role
            self.lineEdit().setText(text)
            self.lineEdit().setPlaceholderText("any")

        required = False  # a period the step needs (a forecast's): empty stands for nothing, so none is suggested

        def set_periods(self, periods: list[str], bounds: tuple[str, str] | None = None) -> None:
            """The periods to step through; ``bounds``, the dataset's active sample, is what empty stands for. That
            default reads as a value, in the text's own colour, with what it is: "1960Q1 (first period)"."""
            self.periods = list(periods)
            self.bounds = bounds if bounds is not None else ((self.periods[0], self.periods[-1]) if self.periods else None)
            edit = self.lineEdit()
            if self.required:
                edit.setPlaceholderText("choose a period")
            elif self.role in ("start", "end") and self.bounds:
                edit.setPlaceholderText(f"{self.bounds[0]} (first period)" if self.role == "start" else f"{self.bounds[1]} (last period)")
            else:
                edit.setPlaceholderText("any")
            if not self.required and self.role in ("start", "end") and self.bounds:
                from PySide6.QtGui import QPalette

                palette = edit.palette()  # the default in the text's colour: grey placeholder text looked disabled
                palette.setColor(QPalette.ColorRole.PlaceholderText, palette.color(QPalette.ColorRole.Text))
                edit.setPalette(palette)
            self.update()

        def stepEnabled(self) -> Any:
            flags = Qt.QAbstractSpinBox.StepEnabledFlag
            return (flags.StepUpEnabled | flags.StepDownEnabled) if self.periods else flags.StepNone

        def stepBy(self, steps: int) -> None:
            if not self.periods:
                return
            text = self.lineEdit().text().strip()
            if text in self.periods:
                i = self.periods.index(text) + steps
            elif self.role in ("start", "end") and self.bounds:
                # empty: the sample's first or last period, whichever arrow
                edge = self.bounds[0] if self.role == "start" else self.bounds[1]
                i = self.periods.index(edge) if edge in self.periods else (0 if self.role == "start" else len(self.periods) - 1)
            else:
                i = 0 if steps > 0 else len(self.periods) - 1
            self.lineEdit().setText(self.periods[max(0, min(len(self.periods) - 1, i))])

    _SPIN["cls"] = PeriodSpin
    return PeriodSpin


def focus_first_field(dialog: Any) -> Any:
    """Give the keyboard to the dialog's first visible field that takes it (in the form's order), as a dialog opens:
    it had gone to the form's scroll area, an extra Tab stop before the first field. Returns the widget, or None."""
    from PySide6.QtCore import Qt as QtNS

    Qt = _qt()
    for editor in getattr(dialog, "editors", {}).values():
        outer = getattr(editor, "outer", None)
        if outer is None or not outer.isVisibleTo(dialog):
            continue
        for widget in [outer, *outer.findChildren(Qt.QWidget)]:
            if (widget.isVisibleTo(dialog) and widget.isEnabled() and widget.focusPolicy() & QtNS.FocusPolicy.TabFocus
                    and not isinstance(widget, (Qt.QScrollArea, Qt.QPushButton, Qt.QToolButton))):
                widget.setFocus(QtNS.FocusReason.TabFocusReason)
                return widget
    return None


FORM_SCREEN_SHARE = 0.7  # of the screen's height, at most, for a dialog's fields: room for its buttons


SHOW_CODE_KEY = "dialogs/show_code"  # whether dialogs open with their code shown (an expert's choice, remembered)


def _desktop_settings(QSettings: Any) -> Any:
    """The app's own settings file (pymetrix/desktop), as the window's layout uses: ``QSettings()`` alone would name
    no organisation and write elsewhere."""
    return QSettings(QSettings.defaultFormat(), QSettings.Scope.UserScope, "pymetrix", "desktop")


def _code_panel(dialog: Any, root: Any, Qt: Any, command_cls: Any, values: Any, after_refresh: list[Any],
                session: Any = None) -> None:
    """**Show the code** under a dialog's fields: the Python the step will record, as the fields are filled in
    (folded by default; opened once, it stays open in every dialog). Read-only: the fields are the way to change it;
    Copy takes it to the clipboard. ``dialog.code_text()`` gives it; ``dialog.code_toggle`` folds it. A dialog whose
    call is built from the session (Check model: the dataset only if there is one) shows it built from ``session``,
    as the step records it."""
    import contextlib

    from PySide6.QtCore import QSettings
    from PySide6.QtCore import Qt as QtNS

    toggle = Qt.QToolButton()
    toggle.setText("Show the code")
    toggle.setCheckable(True)
    toggle.setToolButtonStyle(QtNS.ToolButtonStyle.ToolButtonTextBesideIcon)
    toggle.setArrowType(QtNS.ArrowType.RightArrow)
    toggle.setAutoRaise(True)
    toggle.setToolTip("the Python this step records, as in History and the script (for experts)")
    box = Qt.QPlainTextEdit()
    box.setReadOnly(True)
    box.setMaximumHeight(110)
    box.setStyleSheet("font-family: monospace;")
    copy = Qt.QPushButton("Copy")
    copy.setToolTip("copy the code")
    row = Qt.QWidget()
    line = Qt.QHBoxLayout(row)
    line.setContentsMargins(0, 0, 0, 0)
    line.addWidget(box, 1)
    line.addWidget(copy, 0, QtNS.AlignmentFlag.AlignTop)
    row.setVisible(False)
    root.addWidget(metrics.spaced(toggle))  # a section's room above it, as above the Advanced section's toggle
    root.addWidget(row)

    def code_text() -> str:
        try:
            cmd = command_cls(**values())
            if session is not None:
                from ..commands.session import _dialog_code

                return _dialog_code(cmd, session)  # the session's own objects, as the step will record it
            return cmd.to_code()
        except Exception:  # noqa: BLE001 - a field not filled in yet: say so, never fail the dialog
            return "# the code appears once the fields it needs are filled in"

    def update() -> None:
        if toggle.isChecked():
            box.setPlainText(code_text())

    def make_room(on: bool) -> None:
        """Open, the code takes its height from the form's share of the screen, so the dialog still opens within the
        screen (a long form scrolls a little sooner); folded, the form has its share back."""
        form = getattr(dialog, "form_scroll", None)
        if form is None:
            return
        from .fit import _available

        area = _available(form)
        taken = (row.sizeHint().height() + max(root.spacing(), 0)) / area.height() if on and area is not None else 0.0
        form.share = max(FORM_SCREEN_SHARE - taken, 0.3)
        form.updateGeometry()
        for layout in (root, *root.findChildren(Qt.QLayout)):  # the dialog's size hints read the new share at once
            layout.invalidate()

    def show(on: bool) -> None:
        toggle.setArrowType(QtNS.ArrowType.DownArrow if on else QtNS.ArrowType.RightArrow)
        toggle.setText("Hide the code" if on else "Show the code")
        row.setVisible(on)
        make_room(on)
        update()
        with contextlib.suppress(Exception):  # a preference not saved is no reason to fail
            _desktop_settings(QSettings).setValue(SHOW_CODE_KEY, on)

    toggle.toggled.connect(show)
    copy.clicked.connect(lambda: Qt.QApplication.clipboard().setText(code_text()))
    after_refresh.append(update)
    try:
        remembered = _desktop_settings(QSettings).value(SHOW_CODE_KEY, False) in (True, "true")
    except Exception:  # noqa: BLE001
        remembered = False
    toggle.setChecked(remembered)
    dialog.code_toggle, dialog.code_text, dialog.code_box = toggle, code_text, box


def command_dialog(command_cls: type[Command], session: Session, parent: Any = None, initial: Command | None = None) -> Any:
    """A dialog generated from the command's fields: radio buttons for short choice lists,
    dropdowns, spin boxes, series checklists, fields shown only when relevant, sections,
    and a settings sub-form for the chosen test. ``initial`` fills it in (reopening a step)."""
    Qt = _qt()

    class StepDialog(Qt.QDialog):  # type: ignore[misc, name-defined]
        running = False
        focused = False

        def showEvent(self, event: Any) -> None:  # noqa: N802 - the first field takes the keys, not the scroll area
            super().showEvent(event)
            if not self.focused:
                self.focused = True
                focus_first_field(self)

        def reject(self) -> None:  # Cancel, Esc or the title bar's close: while the step runs, stop it instead
            if self.running:
                cancel_running()
                return
            super().reject()

    dialog = StepDialog(parent)
    menu = next((i.menu for i in COMMANDS.values() if i.cls is command_cls), None)
    from ..wording import dialog_title

    title = dialog_title(command_cls, menu)  # the menu item's words (less the "…"), unless the table says more
    dialog.setWindowTitle(title)
    root = Qt.QVBoxLayout(dialog)
    metrics.set_margins(root, "dialog_margin")
    body = Qt.QHBoxLayout()
    root.addLayout(body, 1)
    form_side = Qt.QWidget()
    form_side.setMinimumWidth(420)  # the fields' hints ("e.g. kappa=Gamma(0.3, 0.1)") in full
    outer = Qt.QVBoxLayout(form_side)
    outer.setContentsMargins(0, 0, 0, 0)
    from .fit import scroll_area

    form_scroll = scroll_area(form_side, share=FORM_SCREEN_SHARE, keep_minimum=True)  # a long form scrolls on a small screen
    from PySide6.QtCore import Qt as _QtCore

    form_scroll.setFocusPolicy(_QtCore.FocusPolicy.NoFocus)  # not a Tab stop of its own before the first field
    body.addWidget(form_scroll, 1)
    dialog.form_scroll = form_scroll
    from .help_qt import help_view

    pane = help_view()  # context help: this dialog, then what is chosen in it
    pane.setMinimumWidth(440)
    pane.setVisible(False)
    body.addWidget(pane, 1)
    specs = form_spec(command_cls)
    main = _form(Qt)
    by_group: dict[str, Any] = {}  # group_tabs: each group's fields on a tab of their own
    if getattr(command_cls, "group_tabs", False):
        tabs = Qt.QTabWidget()
        pages = [("General", main)]
        for spec in specs:
            if spec.group and not spec.advanced and spec.group not in by_group:
                by_group[spec.group] = _form(Qt)
                pages.append((spec.group, by_group[spec.group]))
        for title, form in pages:
            page = Qt.QWidget()
            page.setLayout(form)
            tabs.addTab(page, title)
        outer.addWidget(tabs)
        dialog.tabs = tabs
    else:
        outer.addLayout(main)
    editors: dict[str, _Editor] = {}
    rows: dict[str, tuple[Any, int]] = {}
    from ..forms import missing_fields, missing_message, needs_value
    from .dialog_checks import EmptyPicker, OkGate, error_box, most_recent_model

    pickers: dict[str, Any] = {}  # object fields: their line for when there is nothing to pick
    picker_rows: dict[str, tuple[Any, int]] = {}
    picker_empty: dict[str, bool] = {}  # whether each picker has nothing to pick (its line shows)

    def open_instead(cls: Any, initial_name: Any = None) -> None:
        """A picker's button: close this dialog, and the window opens the one that makes what it needs."""
        dialog.open_instead = (cls, initial_name)
        dialog.reject()
    headings: dict[tuple[int, str, bool], tuple[Any, int]] = {}
    advanced: _AdvancedSection | None = None
    shown: dict[str, bool] = {}  # each field's own show_if, open or closed section alike
    # a sub-form's advanced settings (Newton's line search...) go with its group's advanced fields when there are
    # some, after the last, named by the choice they belong to; otherwise at the section's end, under its own label
    with_group = {[s.name for s in specs if s.advanced and s.group == o.group][-1]: o for o in specs
                  if o.kind == "options" and not o.advanced and o.group and any(s.advanced and s.group == o.group for s in specs)}
    sub_labels: dict[str, Any] = {}
    for spec in [*(s for s in specs if not s.advanced), *(s for s in specs if s.advanced)]:  # advanced ones last
        form = main if spec.advanced else by_group.get(spec.group, main)
        if spec.advanced and advanced is None:
            advanced = _AdvancedSection(Qt, main, dialog, command_cls.__name__)
        if spec.group and spec.group not in by_group and (id(form), spec.group, spec.advanced) not in headings:
            if spec.label.lower() == spec.group.lower():  # the field names its own section: no heading above it
                headings[(id(form), spec.group, spec.advanced)] = (form, -1)
            else:
                heading = Qt.QLabel(f"<b>{spec.group}</b>")
                if form.rowCount():  # the same room above every section's heading (metrics), none at the form's top
                    metrics.space_above(heading)
                form.addRow(heading)
                headings[(id(form), spec.group, spec.advanced)] = (form, form.rowCount() - 1)
                if spec.advanced:
                    group = spec.group
                    advanced.add(form.rowCount() - 1, lambda g=group: any(shown.get(s.name) for s in specs if s.advanced and s.group == g)
                                 or any(shown.get(o.name) and editors[o.name].advanced_form.rowCount() > 0
                                        for o in with_group.values() if o.group == g))
        editors[spec.name] = _Editor(Qt, spec)
        form.addRow(spec.label + (" *" if needs_value(spec) else ""), editors[spec.name].outer)  # * : OK waits for it
        _python_hint_on_label(form, editors[spec.name])
        rows[spec.name] = (form, form.rowCount() - 1)
        if spec.advanced:
            advanced.add(form.rowCount() - 1, lambda n=spec.name: shown.get(n))
        if spec.kind == "object" and spec.source and spec.dynamic:  # nothing to pick: say so, and where to make one
            pickers[spec.name] = EmptyPicker(spec, session, open_instead)
            form.addRow("", pickers[spec.name].widget)
            picker_rows[spec.name] = (form, form.rowCount() - 1)
            if spec.advanced:  # in the Advanced section: shown with it, when its field is and there is nothing to pick
                advanced.add(form.rowCount() - 1, lambda n=spec.name: shown.get(n) and picker_empty.get(n, False))
        if spec.name in with_group:  # the chosen solver's (method's...) own settings, with the group's other settings
            sub, part = with_group[spec.name], editors[with_group[spec.name].name]
            sub_labels[sub.name] = Qt.QLabel(sub.label)
            form.addRow(sub_labels[sub.name], part.advanced_part)
            advanced.add(form.rowCount() - 1, lambda n=sub.name, e=part: shown.get(n) and e.advanced_form.rowCount() > 0)
    for spec in specs:  # a sub-form's advanced settings: in the same section, under the sub-form's label
        if spec.kind == "options" and spec.name not in sub_labels:
            if advanced is None:
                advanced = _AdvancedSection(Qt, main, dialog, command_cls.__name__)
            part = editors[spec.name]
            main.addRow(spec.label, part.advanced_part)
            advanced.add(main.rowCount() - 1, lambda n=spec.name, e=part: shown.get(n) and e.advanced_form.rowCount() > 0)

    busy = False
    import dataclasses as _dc

    carried = ({f.name: getattr(initial, f.name) for f in _dc.fields(command_cls) if f.name not in editors and f.init}
               if initial is not None and _dc.is_dataclass(initial) else {})  # hidden fields, set in code: kept

    def values() -> dict[str, Any]:
        return {**carried, **{k: e.get() for k, e in editors.items()}}

    def refresh() -> None:
        nonlocal busy
        if busy:
            return
        busy = True
        try:
            for _ in range(2):  # a second pass settles fields that depend on fields placed after them
                for spec in specs:
                    editor = editors[spec.name]
                    if spec.dynamic and spec.kind == "series":
                        current = values()
                        editor.set_options(session.choices(spec, current, scoped=not editor.show_all))
                        editor.set_scope(_scope_note(session, current.get(spec.depends_on or "", "")))
                    elif spec.dynamic:
                        editor.set_options(session.choices(spec, values()))
                    if spec.kind == "options":
                        current = values()
                        editor.rebuild(session.options_form(spec, current), current.get(spec.depends_on))
                        if spec.name in sub_labels:  # named by the choice: "Newton", "Gauss-Seidel"
                            sub_labels[spec.name].setText(str(current.get(spec.depends_on) or spec.label))
            current = values()
            periods = session.period_choices(current)
            bounds = session.sample_bounds(current)
            for spec in specs:
                if spec.kind == "period":
                    editors[spec.name].widget.set_periods(periods, bounds)
            shown.update({spec.name: spec.visible(current) for spec in specs})
            for spec in specs:
                if not spec.advanced:
                    form, row = rows[spec.name]
                    form.setRowVisible(row, shown[spec.name] and _has_main_rows(editors[spec.name]))
            for name, picker in pickers.items():  # an object field with nothing to pick: a line saying so
                form, row = picker_rows[name]
                editor = editors[name]
                options = ([editor.widget.item(i).text() for i in range(editor.widget.count())] if editor.spec.multiple
                           else [editor.widget.itemText(i) for i in range(editor.widget.count())
                                 if editor.widget.itemText(i) != editor._blank()])
                picker_empty[name] = picker.update(options)
                if not editor.spec.advanced:  # an advanced one shows with its section (advanced.sync, below)
                    form.setRowVisible(row, shown[name] and picker_empty[name])
            for (form_id, group, in_advanced), (form, row) in headings.items():
                if row < 0 or in_advanced:
                    continue
                form.setRowVisible(row, any(shown[s.name] for s in specs if s.group == group and not s.advanced
                                            and id(rows[s.name][0]) == form_id))
            if advanced is not None:
                advanced.sync()
        finally:
            busy = False
        check_span()
        check_required()
        update_help()
        for hook in after_refresh:
            hook()

    after_refresh: list[Any] = []  # the field labels' access keys, once the buttons (whose keys they avoid) exist
    ok_button: list[Any] = []
    gate_box: list[Any] = []  # the OkGate, once the buttons exist

    def missing_now(current: dict[str, Any]) -> tuple[str, list[FieldSpec]]:
        """What OK waits for, by the fields' display labels; a field in the Advanced settings section is named as
        such while the section is closed ("Choose Lag order (in Advanced settings)...")."""
        missing = missing_fields(specs, current)
        closed = advanced is not None and not advanced.expanded
        named = [_dc.replace(s, label=f"{s.label} (in {_AdvancedSection.TEXT})") if s.advanced and closed else s
                 for s in missing]
        return missing_message(named), missing

    def check_required() -> None:
        """Hold OK while a field marked * is empty, and say which under the form."""
        if gate_box:
            gate_box[0].set("required", missing_now(values())[0])

    def check_span() -> None:
        """Start after end: mark End and hold OK until it is fixed."""
        if "start" not in editors or "end" not in editors:
            return
        from types import SimpleNamespace

        from ..commands.session import span_errors

        problems = span_errors(SimpleNamespace(start=editors["start"].get(), end=editors["end"].get()))
        end_widget = editors["end"].widget
        # marked by its text's colour, not a style-sheet border (which replaced the spin box's native arrows with a box
        # glyph); the reason is in the line above the buttons, its tooltip and its accessible description
        from PySide6.QtGui import QPalette

        if not hasattr(end_widget, "_own_palette"):
            end_widget._own_palette = QPalette(end_widget.palette())
        marked = QPalette(end_widget._own_palette)
        if problems:
            marked.setColor(QPalette.ColorRole.Text, theme.colour("error"))
        end_widget.setPalette(marked)
        end_widget.setProperty("invalid", bool(problems))
        end_widget.setAccessibleDescription(problems[0] if problems else "")
        from ..wording import python_hint

        end_spec = editors["end"].spec  # the field's tooltip keeps its Python hint ("Python: end="), as _choice_tooltips set it
        plain = f"{end_spec.help}\n{python_hint(end_spec.name)}" if end_spec.help else python_hint(end_spec.name)
        end_widget.setToolTip(f"{problems[0]}\n{plain}" if problems else plain)
        if gate_box:
            gate_box[0].set("span", problems[0] if problems else None)

    def update_help() -> None:
        """Follow the dialog's choices, unless the reader has followed a link elsewhere."""
        if not pane.isVisible() or pane.current() not in ("", "context"):
            return
        from ..help import dialog_markdown

        text = dialog_markdown(command_cls, {k: e.get() for k, e in editors.items()})
        if pane.history and pane.history[-1] == ("context", text):
            return
        if pane.history and pane.history[-1][0] == "context":
            pane.history.pop()
        pane.show_markdown(text, "context")

    pane.on_change.append(lambda: update_help() if pane.current() == "context" else None)  # Back: refresh the context

    def toggle_help() -> None:
        pane.setVisible(not pane.isVisible())
        if pane.isVisible():
            pane.history.clear()
            update_help()

    if initial is not None:
        for name, editor in editors.items():
            editor.set(getattr(initial, name))
    for editor in editors.values():
        editor.connect(refresh)
        editor.on_scope = refresh
        if isinstance(editor.widget, Qt.QComboBox) and editor.widget.isEditable():  # a name typed, not picked: OK follows
            editor.widget.editTextChanged.connect(lambda *_: check_required())
    refresh()
    if initial is None:  # two fields picking the same kind of object (a comparison) start on different ones
        seen: dict[str, list[str]] = {}
        for spec in specs:
            if spec.kind != "object" or not spec.source or spec.multiple:
                continue
            taken = seen.setdefault(spec.source, [])
            value = editors[spec.name].get()
            if value in taken:
                free = [c for c in session.choices(spec, values()) if c and c not in taken]
                if free:
                    editors[spec.name].set(free[0])
                    value = free[0]
            taken.append(value)
        for spec in specs:  # a list of equations or models left empty: the newest whole model (a template just made)
            if spec.kind == "object" and spec.multiple and needs_value(spec) and not editors[spec.name].get():
                newest = most_recent_model(session, spec.source or "")
                if newest:
                    editors[spec.name].set(newest)
        refresh()

    from ..plugins import plugin_of_command

    origin = plugin_of_command(command_cls)
    if origin is not None:  # not pymetrix's own: say whose
        mark = Qt.QLabel(f"From the plugin {origin.name} {origin.version}" + (f" ({origin.license})" if origin.license else ""))
        mark.setStyleSheet("color: palette(link-visited);")  # purple, or its dark-theme match
        mark.setToolTip(origin.description or origin.name)
        root.insertWidget(0, mark)
        dialog.plugin_mark = mark
    _data_preview(dialog, editors, root, Qt)  # Load data: the file's first rows, and its time and units guessed
    _family_hint(dialog, editors, root, Qt, session)  # Estimate: what the dependent variable looks like
    std = Qt.QDialogButtonBox.StandardButton
    buttons = Qt.QDialogButtonBox(std.Ok | std.Cancel | std.Help)
    needed = Qt.QLabel()  # what OK waits for, above the buttons
    needed.setWordWrap(True)
    needed.setStyleSheet(theme.muted_css())
    needed.setVisible(False)
    root.addWidget(needed)
    _code_panel(dialog, root, Qt, command_cls, values, after_refresh, session)  # the Python the step records: for experts, folded
    root.addWidget(buttons)
    ok_button.append(buttons.button(std.Ok))
    gate = OkGate(ok_button[0], needed)
    gate_box.append(gate)
    dialog.ok_gate, dialog.needed_note, dialog.pickers = gate, needed, pickers
    check_span()
    check_required()
    _live_expression(dialog, editors, root, Qt, session, command_cls, gate.proxy("expression"))  # Matrix > Expression: checked as typed
    from .formula_check import live_formulas

    live_formulas(dialog, command_cls, editors, specs, gate, session, values)  # formulas checked as typed, under each
    from .form_keys import assign_keys

    def keys() -> None:  # Alt+letter to each field: unique letters, none of the buttons' (form_keys)
        dialog.access_keys = assign_keys(dialog, editors, {n: f for n, (f, _) in rows.items()}, buttons, advanced)

    after_refresh.append(keys)
    keys()
    _rows_preview(dialog, editors, root, Qt, session, command_cls, gate.proxy("preview"), values)  # Generate series for each: the rows
    buttons.helpRequested.connect(toggle_help)
    from PySide6.QtGui import QKeySequence, QShortcut

    QShortcut(QKeySequence(QKeySequence.StandardKey.HelpContents), dialog, activated=toggle_help)
    dialog.help_pane, dialog.toggle_help = pane, toggle_help
    dialog.result_value = None
    dialog.command = None
    dialog.editors = editors  # for tests and scripting
    dialog.row_visible = lambda name: shown[name] if editors[name].spec.advanced else rows[name][0].isRowVisible(rows[name][1])
    dialog.row_label = lambda name: rows[name][0].labelForField(editors[name].outer)  # its label, with * when needed
    _advanced_presentation(dialog, advanced, specs, editors, initial, refresh, shown)

    def accept() -> None:
        from ..progress import Cancelled

        if dialog.running or _BUSY["running"]:  # OK pressed again, or another step is still running
            return
        current = values()
        missing, fields = missing_now(current)
        bad = advanced.update_header()[1] if advanced is not None else 0  # an advanced setting that is not valid
        if missing or gate.held() or bad:  # never run (or record) a step whose call cannot be built: say why, here
            if any(f.advanced for f in fields) or bad:  # the empty (or not valid) field is in the closed section: open it
                dialog.show_advanced(True)
                missing = missing_now(current)[0]
            gate.set("required", missing or (advanced.problem() if bad else None))
            return
        try:
            dialog.command = command_cls(**current)
            if type(dialog.command).__name__ == "RunCode":  # typed code may draw in the window: on its own thread
                dialog.result_value = session.run(dialog.command)
            else:  # the window stays live, with progress and Cancel for a long step
                dialog.running = True
                gate.hold("running", "the step is running")
                try:
                    dialog.result_value = run_blocking(lambda: session.run(dialog.command), dialog.windowTitle())
                finally:
                    dialog.running = False
                    gate.release("running")
        except Cancelled:
            Qt.QMessageBox.information(dialog, "Cancelled", "The step was cancelled; nothing was recorded.")
            return
        except Exception as exc:  # in plain words, titled after the dialog; the technical text under Details
            dialog.last_error = error_box(dialog, dialog.windowTitle(), exc, current)
            return
        if _BUSY.pop("cancel ignored", False) and not getattr(session, "recalculation_stopped", False):
            Qt.QMessageBox.information(dialog, "Finished before stopping", "The step finished before it could stop "
                                       "(it has no point at which to stop), so it was recorded.")
        elif getattr(session, "recalculation_stopped", False):
            Qt.QMessageBox.information(dialog, "Recalculation stopped", "The step was recorded, but the recalculation "
                                       "after it was cancelled: the results that depend on it are out of date (Problems "
                                       "lists them; Recalculate brings them up to date).")
        dialog.accept()

    buttons.accepted.connect(accept)
    buttons.rejected.connect(dialog.reject)
    return dialog



def _live_expression(dialog: Any, editors: dict[str, Any], root: Any, Qt: Any, session: Any, command_cls: Any, ok: Any) -> None:
    """Under a dialog whose command checks an expression as it is typed (``live_expression``, the field's name: Model >
    Matrix > Expression…): a moment (300 ms) after typing stops, the problem (an unknown name with a suggestion, a
    construct not allowed, a parse error with its column), with OK disabled until it is fixed; else, when it evaluates
    cheaply, what it gives (``pymetrix.matrix_expr.preview``)."""
    field = getattr(command_cls, "live_expression", "")
    if not field or field not in editors:
        return
    from PySide6.QtCore import QTimer

    from ..commands.matrices import live_check

    label = Qt.QLabel(" ")
    label.setWordWrap(True)
    from PySide6.QtCore import Qt as QtNS

    label.setTextInteractionFlags(QtNS.TextInteractionFlag.TextSelectableByMouse)
    root.insertWidget(root.count() - 1, label)
    timer = QTimer(dialog)
    timer.setSingleShot(True)
    timer.setInterval(300)

    def update() -> None:
        found = live_check(str(editors[field].get() or ""), str(editors["name"].get() or "") if "name" in editors else "M", session)
        if found.problem:
            label.setText(found.problem)
            label.setStyleSheet(theme.css("error"))
        else:
            text = found.what or found.skipped or "No problems"
            if found.rows or found.cols:
                text += f"; rows {_short_names(found.rows)}; columns {_short_names(found.cols)}"
            label.setText(text)
            label.setStyleSheet(theme.css("success") if found.what else theme.muted_css())
        ok.setEnabled(not found.problem)
        ok.setToolTip(found.problem)

    timer.timeout.connect(update)
    for name in (field, "name"):
        if name in editors:
            editors[name].connect(timer.start)
    dialog.expression_check, dialog.expression_timer = label, timer
    dialog.check_expression = update  # for tests: check now, without waiting
    update()


def _rows_preview(dialog: Any, editors: dict[str, Any], root: Any, Qt: Any, session: Any, command_cls: Any, ok: Any,
                  values: Any) -> None:
    """Under a dialog whose command previews what it will make (``preview_rows``: Generate series with For each): a
    table of the rows, problems in red, replacements in amber, and a summary; OK is held while there are problems. It
    follows the fields a moment (300 ms) after typing stops."""
    if not hasattr(command_cls, "preview_rows"):
        return
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QBrush

    note = Qt.QLabel()
    note.setWordWrap(True)
    table = Qt.QTableWidget(0, 0)
    table.setMaximumHeight(220)
    table.setEditTriggers(Qt.QAbstractItemView.EditTrigger.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    root.insertWidget(root.count() - 1, note)
    root.insertWidget(root.count() - 1, table)
    colours = {"error": QBrush(theme.colour("error")), "warning": QBrush(theme.colour("warning"))}

    def update() -> None:
        try:
            found = command_cls(**values()).preview_rows(session)
        except Exception as exc:  # noqa: BLE001 - say what is wrong, and hold OK
            found = {"headings": [], "rows": [], "marks": [], "summary": str(exc), "ok": False}
        shown = found is not None
        note.setVisible(shown)
        table.setVisible(shown and bool(found["rows"]))
        if not shown:
            ok.setEnabled(True)
            ok.setToolTip("")
            return
        table.setColumnCount(len(found["headings"]))
        table.setHorizontalHeaderLabels(found["headings"])
        table.setRowCount(len(found["rows"]))
        for r, (row, mark) in enumerate(zip(found["rows"], found["marks"])):
            for c, value in enumerate(row):
                item = Qt.QTableWidgetItem(str(value))
                if mark in colours:
                    item.setForeground(colours[mark])
                table.setItem(r, c, item)
        table.resizeColumnsToContents()
        note.setText(("Preview: " if found["ok"] else "") + found["summary"])
        note.setStyleSheet("" if found["ok"] else theme.css("error"))
        ok.setEnabled(found["ok"])
        ok.setToolTip("" if found["ok"] else found["summary"])

    timer = QTimer(dialog)
    timer.setSingleShot(True)
    timer.setInterval(300)
    timer.timeout.connect(update)
    for editor in editors.values():
        editor.connect(timer.start)
    dialog.rows_preview, dialog.rows_note, dialog.refresh_rows_preview = table, note, update  # for tests: update now
    update()


def _short_names(names: Any, most: int = 6) -> str:
    names = list(names)
    return ", ".join(names[:most]) + (f", … ({len(names)})" if len(names) > most else "")


def _data_preview(dialog: Any, editors: dict[str, Any], root: Any, Qt: Any) -> None:
    """Under a dialog that reads a data file (a file field with time, entity and freq fields): the first rows, what
    they suggest, and those guesses in the fields still empty (or still holding an earlier guess)."""
    import os

    if "path" not in editors or editors["path"].spec.kind != "file" or not {"time", "entity", "freq"} <= set(editors):
        return
    note = Qt.QLabel()
    note.setWordWrap(True)
    note.setStyleSheet(theme.muted_css())
    table = Qt.QTableWidget(0, 0)
    table.setMaximumHeight(170)
    table.setEditTriggers(Qt.QAbstractItemView.EditTrigger.NoEditTriggers)
    root.addWidget(note)
    root.addWidget(table)
    guessed: dict[str, Any] = {}

    def update() -> None:
        from ..io.registry import sniff

        path = str(editors["path"].get() or "")
        if not path or not os.path.isfile(path):
            note.setText("Choose a file to see its first rows.")
            table.setVisible(False)
            return
        found = sniff(path)
        head = found.head
        table.setVisible(len(head.columns) > 0)
        table.setColumnCount(len(head.columns))
        table.setRowCount(min(len(head), 8))
        table.setHorizontalHeaderLabels([str(c) for c in head.columns])
        for r in range(table.rowCount()):
            for c in range(len(head.columns)):
                table.setItem(r, c, Qt.QTableWidgetItem(str(head.iat[r, c])))
        table.resizeColumnsToContents()
        note.setText("Preview: " + "; ".join(found.notes) if found.notes else "Preview")
        for field, value in (("time", found.time), ("entity", found.entity), ("freq", found.freq)):
            current = editors[field].get() or ""
            if value and (not current or current == guessed.get(field)):
                editors[field].set(value)
                guessed[field] = value

    editors["path"].connect(update)
    update()
    dialog.preview, dialog.preview_note, dialog.refresh_preview = table, note, update


_FAMILY_HINTS = {"binary": ("logit", "a 0/1 outcome: logit or probit"),
                 "count": ("poisson", "a count (whole numbers from 0): poisson, or negative binomial if it varies more than its mean"),
                 "zeros": ("negative binomial", "a count with many zeros: negative binomial or zero-inflated poisson"),
                 "ordered": ("ordered logit", "a few ordered values: ordered logit or probit"),
                 "censored": ("tobit", "a pile-up at {limit:g} ({share:.0%} of rows): tobit, censored at {limit:g}")}


def outcome_kind(values: Any) -> tuple[str, dict[str, float]] | None:
    """What a dependent variable looks like, for the Estimate dialog's hint: binary, count, zeros (a count with many),
    ordered (a few whole values), censored (a lump at the lowest value), or None (continuous, nothing to suggest)."""
    import numpy as np
    import pandas as pd

    v = pd.Series(values).dropna().to_numpy(float)
    if len(v) < 10:
        return None
    unique = np.unique(v)
    whole = np.allclose(v, np.round(v))
    if set(unique) <= {0.0, 1.0} and len(unique) == 2:
        return "binary", {}
    if whole and 3 <= len(unique) <= 7 and unique.min() >= 0 and len(v) >= 10 * len(unique):
        return "ordered", {}
    positive = v[v > 0]
    if whole and unique.min() >= 0 and len(unique) > 7 and len(positive) and np.median(positive) <= 50:  # counts are small
        zeros = float(np.mean(v == 0))
        return ("zeros", {"share": zeros}) if zeros > 0.3 else ("count", {})
    low = float(unique.min())
    share = float(np.mean(v == low))
    if share > 0.1 and len(unique) > 20:  # a pile-up at the lowest value, the rest spread out
        return "censored", {"limit": low, "share": share}
    return None


def _family_hint(dialog: Any, editors: dict[str, Any], root: Any, Qt: Any, session: Any) -> None:
    """Under the Estimate dialog: what the dependent variable looks like, and the likelihood family that suits it, with
    a button to choose it. A hint only: nothing changes unless the button is pressed."""
    if not {"formula", "family", "method", "data"} <= set(editors):
        return
    row = Qt.QWidget()
    layout = Qt.QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    text = Qt.QLabel()
    text.setWordWrap(True)
    text.setStyleSheet(theme.muted_css())
    use = Qt.QPushButton("Use it")
    layout.addWidget(text, 1)
    layout.addWidget(use)
    root.addWidget(row)
    state: dict[str, Any] = {}

    def update() -> None:
        from ..estimation import Equation
        from ..expr import evaluate

        state.clear()
        row.setVisible(False)
        data = session.objects.get(editors["data"].get())
        formula = str(editors["formula"].get() or "")
        if data is None or "~" not in formula:
            return
        try:
            lhs = Equation(formula).lhs
            values = evaluate(lhs, data.lagged)
        except Exception:  # noqa: BLE001 - a formula being typed: no hint until it reads
            return
        kind = outcome_kind(values)
        if kind is None:
            return
        family, message = _FAMILY_HINTS[kind[0]]
        info = kind[1]
        current = editors["family"].get()
        if current not in ("gaussian", None, "") and current != family:
            return  # a family is chosen already: leave it
        text.setText(f"{lhs} looks like " + message.format(**info) + ".")
        state.update(family=family, limit=info.get("limit"))
        use.setVisible(current != family)
        row.setVisible(True)

    def choose() -> None:
        if not state:
            return
        method = editors["method"]
        if method.get() != "ML":
            method.set("ML")
        editors["family"].set(state["family"])
        if state["family"] == "tobit" and "censor_left" in editors and state.get("limit") is not None:
            editors["censor_left"].set(f"{state['limit']:g}")
        update()

    use.clicked.connect(choose)
    for name in ("formula", "data", "family"):
        editors[name].connect(update)
    update()
    dialog.family_hint, dialog.family_hint_button, dialog.refresh_family_hint = text, use, update


_KINDS = {
    "Dataset": "Dataset",
    "Solution": "Solution",
    "Equation": "Equation",
    "FittedEquation": "Fitted equation",
    "FittedBayesEquation": "Bayesian equation",
    "TestResult": "Test result",
    "FittedVAR": "Fitted VAR",  # as "Fitted BVAR", "Fitted equation"
    "FittedVECM": "Fitted VECM",
    "IRF": "Impulse responses",
    "FEVD": "Variance decomposition",
    "RollingFit": "Rolling estimates",
    "JohansenResult": "Johansen test",
    "LagSelection": "Lag selection",
    "OrderSelection": "ARMA order selection",
    "Selection": "Selection of series",
    "Figure": "Figure",
    "FittedStateSpace": "State space",
    "FittedLocalProjection": "Local projections",
    "ndarray": "Matrix (NumPy)",
    "DataFrame": "Table (pandas)",
    "Series": "Series (pandas)",
    "float": "Number",
    "int": "Number",
    "float64": "Number",
    "str": "Text",
    "list": "List",
    "dict": "Dictionary",
    "Results": "Results by series",
    "Sample": "Sample",
    "Template": "Template model",
    "DynareModel": "Dynare model",
    "Model": "Model",
    "LinearSolution": "Rational-expectations solution",
    "FittedModel": "Estimated model",
    "Posterior": "Posterior (Metropolis-Hastings)",
    "Identification": "Identification",
    "OccBinPath": "OccBin path",
    "SecondOrderSolution": "Second-order solution",
    "ThirdOrderSolution": "Third-order solution",
    "ParticleFilter": "Particle filter",
    "OptimalPolicy": "Optimal policy",
    "OSRResult": "Optimal simple rule",
    "SweepResult": "Parameter sweep",
    "ScenarioSet": "Scenarios",
    "ForecastRound": "Forecast round",
    "Backtest": "Backtest",
    "Decomposition": "Shock decomposition",
    "Table": "Table",
}


def kind_of(obj: Any) -> str:
    """What an object is, in words: ``FittedModel`` -> "Estimated model"; others by splitting the class name.
    A saved result says what it was: "Saved fitted equation"."""
    from ..saved import SavedResult

    if isinstance(obj, SavedResult):
        inner = _kind_words(obj.kind or "Result")
        return "Saved " + (inner if inner[:2].isupper() else inner[0].lower() + inner[1:])  # "Saved VAR", "Saved fitted equation"
    return _kind_words(type(obj).__name__)


def _kind_words(name: str) -> str:
    import re

    if name in _KINDS:
        return _KINDS[name]
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name).split()
    return " ".join([words[0]] + [w.lower() if not w.isupper() else w for w in words[1:]]) if words else name


def window_title(name: str, obj: Any) -> str:
    """An object window's title: its name and kind, and a dataset's own title when it has one."""
    title = getattr(getattr(obj, "meta", None), "title", "") or ""
    return f"{name} — {kind_of(obj)}" + (f": {title}" if title and title != name else "")


def _cell(value: Any, p_value: bool = False) -> str:
    """A table cell's text: four decimals (scientific beyond), and a p-value below 0.0001 as "<0.0001" (on screen
    only: every copy gives the number itself)."""
    from ..output import _fmt

    if p_value and _is_number(value) and 0 <= value < 1e-4:
        return "<0.0001"
    return _fmt(value)


def _number_type(hint: Any) -> tuple[type, bool]:
    """A number field's own type (int or float) and whether it may be empty (``int | None``)."""
    import types
    import typing

    if isinstance(hint, types.UnionType) or typing.get_origin(hint) is typing.Union:
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        return (int if args == [int] else float), len(args) < len(typing.get_args(hint))
    return (int if hint is int else float), False


def short_placeholder(help_text: str, most: int = 40) -> str:
    """A text field's grey hint, short enough for a 300-pixel field: the help's example ("e.g. recession == 0"), else
    its first clause, shortened at a word ("empty for th…" was cut mid-word). The full help is the field's tooltip."""
    import re

    text = (help_text or "").strip()
    example = re.search(r"\b(?:e\.g\.|for example|such as)\s*,?\s*(.+?)(?:;|: |$)", text)
    if example:
        found = example.group(1).strip().rstrip(".,")
        found = re.split(r"\s+or\s+(?=[A-Za-z_]\w*\s*(?:[=<>!]))", found)[0].strip()  # the first of several examples
        while found.count("(") > found.count(")"):  # a parenthesis the example opens and the clause does not close
            found = found[:found.rindex("(")].rstrip()
        if found and len(found) <= most:
            return f"e.g. {found}"
    first = re.split(r";|: | \(", text)[0].strip().rstrip(".,")
    if len(first) <= most:
        return first
    return first[:most].rsplit(" ", 1)[0].rstrip(",:;") + "…"


def _empty_means(help_text: str) -> str:
    """What leaving a number empty stands for, from its help ("empty = automatic" -> automatic)."""
    import re

    found = re.search(r"\bempty = ([^,;]+)", help_text or "")
    return re.sub(r"\s*\(.*$", "", found.group(1)).strip() if found else ""


def _zero_means(help_text: str) -> str:
    """What 0 stands for in a number field, from its help ("0 = automatic (in parallel when it pays)" -> automatic)."""
    import re

    found = re.search(r"\b0 = ([^,;]+)", help_text or "")
    return re.sub(r"\s*\(.*$", "", found.group(1)).strip() if found else ""


def _is_number(value: Any) -> bool:
    import numpy as np

    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def _is_p_column(name: str) -> bool:
    """p>|t|, p>|z|, p-value, prob: the columns whose tiny values read better as <0.0001."""
    n = name.strip().lower()
    return n in ("p", "p-value", "p value", "pvalue", "prob", "prob.", "p-val") or n.startswith(("p>", "p(", "p-value"))


def _fill_grid(Qt: Any, grid: Any, frame: Any, table: Any) -> None:
    """A result table in a grid: numbers right-aligned (headers too) and kept with each cell for copying, text left;
    compact rows; the row numbers hidden when the first column already names each row."""
    from PySide6.QtCore import Qt as QtNS

    from ..summary_layout import is_numberish

    right = QtNS.AlignmentFlag.AlignRight | QtNS.AlignmentFlag.AlignVCenter
    p_columns = {c for c, name in enumerate(frame.columns) if _is_p_column(str(name))}
    numeric_columns = set()
    for c in range(len(frame.columns)):
        values = [frame.iat[r, c] for r in range(len(frame))]
        present = [v for v in values if v is not None and not (isinstance(v, float) and v != v)]
        if present and all(is_numberish(v) for v in present):  # numbers, or text that reads as one (a test's df)
            numeric_columns.add(c)
    for r in range(len(frame)):
        for c in range(len(frame.columns)):
            value = frame.iat[r, c]
            item = Qt.QTableWidgetItem(_cell(value, c in p_columns))
            if c in numeric_columns:  # a column of numbers right-aligned; a number among text keeps the text's side
                item.setTextAlignment(right)
            if _is_number(value):
                import numpy as np

                whole = isinstance(value, (int, np.integer))
                item.setData(QtNS.ItemDataRole.UserRole, int(value) if whole else float(value))
            grid.setItem(r, c, item)
    for c in numeric_columns:
        header = grid.horizontalHeaderItem(c)
        if header is not None:
            header.setTextAlignment(right)
    for c in range(len(frame.columns)):  # a header shortened by a narrow column reads in full in its tooltip
        header = grid.horizontalHeaderItem(c)
        if header is not None:
            header.setToolTip(header.text())
    if table.index is None and len(frame.columns) and len(frame) and all(isinstance(frame.iat[r, 0], str) for r in range(len(frame))):
        grid.verticalHeader().setVisible(False)  # 1, 2, 3 add nothing when the first column names the rows
    grid.setAlternatingRowColors(True)  # every other row shaded, in the palette's alternate base colour
    header_font = grid.horizontalHeader().font()
    header_font.setBold(True)
    grid.horizontalHeader().setFont(header_font)
    # a column widened by hand: the frame follows (not each column _size_columns sets: once a column, on a table of
    # hundreds of columns, that was hundreds of row fits, half a minute for a 200-equation model's solution)
    grid.horizontalHeader().sectionResized.connect(lambda *_: None if getattr(grid, "_sizing", False) else _fit_height(grid))
    _size_columns(grid)
    grid.resizeRowsToContents()
    from PySide6.QtCore import QEvent, QObject

    class _Fit(QObject):  # sized when shown: rows and headers have their final heights only once laid out
        def eventFilter(self, watched: Any, event: Any) -> bool:
            if event.type() == QEvent.Type.Show and not getattr(watched, "_sized_shown", False):
                watched._sized_shown = True  # the columns again, now the header has its style's font and margins:
                _size_columns(watched)  # sized before, "imaginary", "modulus" and "converged" headers were cut
            if event.type() in (QEvent.Type.Show, QEvent.Type.Resize):  # the width decides whether a scroll bar needs room
                _fit_height(watched)
            return False

    grid._fit = _Fit(grid)
    grid.installEventFilter(grid._fit)


_WIDEST = 520  # pixels: a wider column wraps its text instead of pushing the table off the window


_CELL_PAD = 10  # pixels added to each column, so the numbers do not run into the grid lines


def _size_columns(grid: Any) -> None:
    """Each column as wide as its values and its header (whichever is wider), padded, at most ``_WIDEST``. The rows
    are fitted once, after (``grid._sizing`` holds off the fit each column's new width would otherwise ask for)."""
    grid._sizing = True
    try:
        grid.resizeColumnsToContents()
        header = grid.horizontalHeader()
        for c in range(grid.columnCount()):
            grid.setColumnWidth(c, grid.columnWidth(c) + _CELL_PAD)
            item = grid.horizontalHeaderItem(c)
            if item is not None and item.text():  # the header's own text, measured in the header's font, with its margins
                margin = 2 * grid.style().pixelMetric(_qt().QStyle.PixelMetric.PM_HeaderMargin, None, header) + 6
                wanted = header.fontMetrics().horizontalAdvance(item.text()) + margin
                if grid.columnWidth(c) < wanted:
                    grid.setColumnWidth(c, wanted)
            if grid.columnWidth(c) > _WIDEST:
                grid.setColumnWidth(c, _WIDEST)
    finally:
        grid._sizing = False
    _fit_height(grid)


def _fit_height(grid: Any) -> None:
    """A table no taller than its rows, so the notes under it follow it rather than sit at the window's foot, and no
    wider than its columns."""
    grid.resizeRowsToContents()
    rows = grid.verticalHeader().length()
    header = grid.horizontalHeader().height() if grid.horizontalHeader().isVisible() else 0
    wide = grid.horizontalHeader().length() + (grid.verticalHeader().width() if grid.verticalHeader().isVisible() else 0)
    scroll = grid.horizontalScrollBar().sizeHint().height() if wide > grid.width() - 2 * grid.frameWidth() else 0
    grid.setMaximumHeight(rows + header + 2 * grid.frameWidth() + scroll + 1)
    side = grid.verticalScrollBar().sizeHint().width() if grid.verticalScrollBar().isVisible() else 0
    grid.setMaximumWidth(wide + 2 * grid.frameWidth() + side + 1)  # as wide as its columns, not the window's width


def _navigation_toolbar(canvas: Any, parent: Any) -> Any:
    """matplotlib's toolbar for looking around a figure: Home (as drawn), Back, Forward, Pan and Zoom (to a rectangle).
    Saving and copying are the window's own buttons, so the toolbar's Save, Subplots and Customize are left out, and
    so is its running x, y readout: a click on a series identifies it instead (``chart_click``), and the window's
    Values button shows the value under the mouse for those who want it."""
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

    items = [t for t in NavigationToolbar2QT.toolitems if t[0] in ("Home", "Back", "Forward", None, "Pan", "Zoom")]
    while items and items[-1][0] is None:  # no separator left dangling where Save was
        items.pop()

    class Toolbar(NavigationToolbar2QT):
        toolitems = items

    toolbar = Toolbar(canvas, parent, coordinates=False)
    toolbar.setIconSize(toolbar.iconSize() * 0.8)
    return toolbar


def style_steps_dialog(figure: Any, parent: Any = None) -> Any:
    """The style calls recorded on ``figure``, in order: remove them, move them up or down, or pick one to change.

    After ``exec()``: ``dialog.order()`` is the steps kept, in their new order (original positions), and
    ``dialog.change`` the original position of the step to change (Change…), or None."""
    Qt = _qt()
    from PySide6.QtCore import Qt as QtNS

    Role = QtNS.ItemDataRole
    dialog = Qt.QDialog(parent)
    dialog.setWindowTitle("Recorded styles")
    layout = Qt.QVBoxLayout(dialog)
    metrics.set_margins(layout, "dialog_margin")
    layout.addWidget(Qt.QLabel("Applied in this order, so a later call wins where two set the same thing."))
    steps = Qt.QListWidget()
    for k, code in enumerate(figure.style_steps()):
        item = Qt.QListWidgetItem(code)
        item.setData(Role.UserRole, k)
        steps.addItem(item)
    steps.setCurrentRow(0)
    layout.addWidget(steps)
    dialog.steps = steps
    dialog.change = None

    def move(by: int) -> None:
        row = steps.currentRow()
        if row >= 0 and 0 <= row + by < steps.count():
            steps.insertItem(row + by, steps.takeItem(row))
            steps.setCurrentRow(row + by)

    def remove() -> None:
        if steps.currentRow() >= 0:
            steps.takeItem(steps.currentRow())

    def change() -> None:
        item = steps.currentItem()
        if item is not None:
            dialog.change = item.data(Role.UserRole)
            dialog.accept()

    buttons = Qt.QHBoxLayout()
    dialog.buttons = {}
    for text, action, tip in (("Up", lambda: move(-1), "apply it earlier"), ("Down", lambda: move(1), "apply it later"),
                              ("Remove", remove, "remove the call"), ("Change…", change, "change its keyword arguments")):
        button = Qt.QPushButton(text)
        button.setToolTip(tip)
        button.clicked.connect(action)
        buttons.addWidget(button)
        dialog.buttons[text] = button
    buttons.addStretch(1)
    layout.addLayout(buttons)
    box = Qt.QDialogButtonBox(Qt.QDialogButtonBox.StandardButton.Ok | Qt.QDialogButtonBox.StandardButton.Cancel)
    box.accepted.connect(dialog.accept)
    box.rejected.connect(dialog.reject)
    layout.addWidget(box)
    dialog.order = lambda: [steps.item(i).data(Role.UserRole) for i in range(steps.count())]
    dialog.resize(560, 320)
    return dialog


ARROW_SAY = "press where the arrow starts and drag to where it points"


def figure_view(figure: Any, name: str = "figure", parent: Any = None, on_edit: Any = None, on_place: Any = None,
                on_report: Any = None, on_pick: Any = None, on_restyle: Any = None, on_remove: Any = None,
                on_format: Any = None, on_style: Any = None) -> Any:
    """A figure drawn in the window (resizes with it), with Copy image, Save and Copy data, and Edit… when ``on_edit``
    can change it: ``on_edit("options")`` for titles, shading and marks, ``on_edit("style")`` for its matplotlib style,
    ``on_edit("properties")`` for every property, ``on_edit("note")`` to add a note, ``"arrow"``, ``"textbox"`` and
    ``"change"`` for an arrow, a text box or a change marker. With ``on_place``, Add a note waits for a click on the
    figure and calls ``on_place(panel, x, y)``: the panel, the date as a period (or the number), and the value there.

    With ``on_pick``, the toolbar's Arrow (and Edit… → Draw an arrow, Mark a change) draw by a press and a drag (or
    two clicks), showing the arrow as it is drawn, and call ``on_pick(kind, panel, first, second, series)``: ``kind`` "arrow" or "change", the points as (period or number, value), a change's
    points snapped to the nearest series' data points (``series`` its name). With ``on_restyle(step, props)`` and
    ``on_remove(step)``, what the recorded steps added (notes, arrows, text, lines, bands, change markers) can be
    selected, dragged, edited and deleted on the figure (``chart_edit``); a double-click opens its Format… form
    (``chart_format``), run as ``on_restyle``, and ``on_format(step)`` opens every property of it.

    A right-click (a two-finger click or Ctrl+click on a Mac; the Menu key) opens a menu for what is under the mouse:
    an added item's Format…, words, order and Delete; a series' Format series… and the panel's Format axes… (both
    ``on_style(series, panel)``, the Figure style dialog filled in); drawing an arrow, adding a note there; the
    figure's titles and shading, Copy image and Save. ``page.context_menu(x, y)`` builds it for a point in
    matplotlib's display coordinates."""
    Qt = _qt()
    from .fit import ActionRow, scrolled_page

    page, body = scrolled_page(parent)  # below the figure's readable size, the window scrolls rather than squashes
    layout = Qt.QVBoxLayout(body)
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    except ImportError:
        layout.addWidget(Qt.QLabel("Figures need matplotlib, which PyMetrix requires: pip install matplotlib"))
        return page
    from .chart_look import look, sharpen

    font, surface = look(page)
    canvas = FigureCanvasQTAgg(figure.draw(font=font, surface=surface))  # the app's font and colours; copies stay light
    sharpen(canvas)  # every pixel of a high-resolution screen, the size in inches kept
    canvas.setMinimumSize(320, 220)  # the smallest it is readable at; the window scrolls below it
    page.toolbar = _navigation_toolbar(canvas, body)
    layout.addWidget(page.toolbar)
    layout.addWidget(canvas, 1)
    page.readout = Qt.QLabel(" ")  # what is under the mouse: the nearest point, the bar or the cell, with its period
    page.readout.setStyleSheet(theme.muted_css())
    page.readout.setToolTip("the value under the mouse: the nearest point of a line, a bar or a heatmap cell")
    page.readout.setVisible(False)  # off unless Values is on: a click on a series says what it is
    layout.addWidget(page.readout)

    def hover(event: Any) -> None:
        from ..figures import readout

        if event.inaxes is None:
            page.readout.setText(" ")
            return
        try:
            page.readout.setText(readout(event.inaxes, event.x, event.y))
        except Exception:  # noqa: BLE001 - a readout is a convenience; never let it interrupt the mouse
            page.readout.setText(" ")

    page.hover = hover
    hovering: list[int] = []

    def show_values(on: bool) -> None:
        """Values on the toolbar: the value under the mouse, as it moves, in the line under the figure."""
        while hovering:
            canvas.mpl_disconnect(hovering.pop())
        if on:
            hovering.append(canvas.mpl_connect("motion_notify_event", hover))
        page.readout.setText(" ")
        page.readout.setVisible(on)

    page.toolbar.addSeparator()
    values = page.toolbar.addAction("Show values")
    values.setCheckable(True)
    values.setToolTip("show the value under the mouse as it moves (off: click on a series to see its name, or on a "
                      "point for its period and value)")
    values.toggled.connect(show_values)
    page.values_action = values
    from .chart_click import install

    place_waiting: list[Any] = []  # Add a note's button, once made: while it waits for a click, clicks are for it
    page.picking = None  # Draw an arrow or Mark a change: {"kind", "points"} while it waits for its two clicks

    def free() -> bool:  # clicks are for identifying and editing: no tool, note or pick waiting
        return (not getattr(page.toolbar, "mode", "") and not (place_waiting and place_waiting[0].isChecked())
                and page.picking is None)

    page.editor = None
    if on_restyle is not None and on_remove is not None:
        from .chart_edit import install as editing

        def edit_text(step: int, key: str, old: str) -> None:
            text, ok = Qt.QInputDialog.getText(page, "Edit the words", "Words:", text=str(old))
            if ok and text != old:
                on_restyle(step, {key: text})

        def format_item(step: int) -> None:
            """The Format… form for the step, its changes run as a Change a style step."""
            from .chart_format import fields, format_dialog

            call = figure.style[step]
            if not fields(call):  # nothing the form covers: every property instead
                if on_format is not None:
                    on_format(step)
                return
            dialog = format_dialog(page, call)
            page.format_open = dialog
            if dialog.exec() and dialog.result_props():
                on_restyle(step, dialog.result_props())

        page.format_item = format_item
        page.editor = editing(canvas, figure, on_restyle, on_remove, edit_text, enabled=free, on_format=format_item)
    page.identify = install(canvas, enabled=free, skip=lambda event: page.editor is not None and page.editor.took is event)
    bar = ActionRow()  # under the figure, never over it: what does not fit goes in its » menu
    bar.setSpacing(3)  # deliberate, off metrics' grid: the seven buttons side by side in a window at its opening size (680 wide)

    def copy_image() -> None:
        from PySide6.QtGui import QImage

        Qt.QApplication.clipboard().setImage(QImage.fromData(figure.png(dpi=150)))

    def save() -> None:
        if on_edit is not None:  # the Save figure dialog: a size for print or slides, recorded as a step
            on_edit("save")
            return
        path, _ = Qt.QFileDialog.getSaveFileName(page, "Save figure", f"{name}.png", "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if path:
            figure.save(path)

    if on_edit is not None:
        edit = Qt.QPushButton("Edit…")
        edit.setToolTip("change the figure: titles, shading and marks, or its matplotlib style (each recorded as a step)")
        menu = Qt.QMenu(edit)
        page.edit_actions = {}
        for kind, text in (("options", "Titles, shading and marks…"), ("style", "Style (lines, axes, legend, size)…"),
                           ("properties", "All properties…"), ("note", "Add a note…"), ("method", "matplotlib method…"),
                           (None, None), ("styles", "Recorded styles…"), ("copy", "Apply style from…"),
                           ("house", "Use as house style…")):
            if kind is None:
                menu.addSeparator()
                continue
            action = menu.addAction(text)
            action.triggered.connect(lambda _=False, k=kind: on_edit(k))
            page.edit_actions[kind] = action
        menu.addSeparator()  # added by a dialog, or by clicking on the figure
        for kind, text in (("arrow", "Add an arrow…"), ("textbox", "Add a text box…"), ("change", "Mark a change…")):
            action = menu.addAction(text)
            action.triggered.connect(lambda _=False, k=kind: on_edit(k))
            page.edit_actions[kind] = action
        if on_pick is not None:
            for kind, text, say in (("arrow", "Draw an arrow on the figure", ARROW_SAY),
                                    ("change", "Mark a change on the figure", "drag from one point of a series to another")):
                action = menu.addAction(text)
                action.setToolTip(say)
                action.triggered.connect(lambda _=False, k=kind, w=say: start_picking(k, w))
                page.edit_actions[f"draw_{kind}"] = action
        menu.setToolTipsVisible(True)
        edit.setMenu(menu)
        bar.addWidget(edit)
        page.edit = edit
    if on_place is not None:
        place = Qt.QPushButton("Add a note")
        place.setCheckable(True)
        place.setToolTip("then click on the figure where the note goes")
        bar.addWidget(place)
        page.place = place
        place_waiting.append(place)
        place.toggled.connect(lambda on: page.identify.dismiss() if on else None)

        def clicked(event: Any) -> None:
            """A click while Add a note is down: the panel, and the date (a period on a dated axis) and value there."""
            if not place.isChecked() or event.inaxes is None or getattr(page.toolbar, "mode", ""):
                return
            import pandas as pd
            from matplotlib import dates

            axes = event.inaxes  # a panel's axes, or its right-hand twin: both know the panel and its frequency
            if getattr(axes, "name", "") == "3d":  # a click on a 3-D panel turns it; notes go on flat panels
                page.readout.setVisible(True)
                page.readout.setText("notes go on flat panels: click on one of those")
                return
            panel, freq = getattr(axes, "pymetrix_panel", 0), getattr(axes, "pymetrix_freq", None)
            when = dates.num2date(event.xdata).replace(tzinfo=None)
            x: Any = str(pd.Period(when, freq=freq)) if freq is not None else f"{event.xdata:.4g}"
            place.setChecked(False)
            on_place(panel, x, float(event.ydata))

        page.clicked = clicked
        canvas.mpl_connect("button_press_event", clicked)

    def start_picking(kind: str, say: str) -> None:
        stop_picking()
        page.identify.dismiss()
        if page.editor is not None:
            page.editor.select(None)
        page.picking = {"kind": kind, "points": [], "press": None, "rubber": None, "say": say}
        page.readout.setVisible(True)
        page.readout.setText(f"{say} (Esc: stop)")
        if kind == "arrow" and page.arrow_action is not None and not page.arrow_action.isChecked():
            page.arrow_action.blockSignals(True)
            page.arrow_action.setChecked(True)
            page.arrow_action.blockSignals(False)

    def stop_picking() -> None:
        picking, page.picking = page.picking, None
        if picking is not None and picking.get("rubber") is not None:
            try:
                picking["rubber"].remove()
            except ValueError:
                pass
            canvas.draw_idle()
        if page.arrow_action is not None and page.arrow_action.isChecked():
            page.arrow_action.blockSignals(True)
            page.arrow_action.setChecked(False)
            page.arrow_action.blockSignals(False)
        page.readout.setText(" ")
        page.readout.setVisible(page.values_action.isChecked())

    def picked_point(event: Any, axes: Any = None) -> tuple[Any, ...] | None:
        """A click as (panel, x in the data's terms, value, series): for a change, the nearest data point of the
        nearest series; else where it is, x as a period on a dated axis. ``axes``: read the point on these axes
        (the first click's), wherever the mouse is."""
        from .chart_click import _name, _nearest
        from .chart_edit import _period

        axes = axes if axes is not None else event.inaxes
        if axes is None or getattr(axes, "name", "") == "3d":
            return None
        panel = getattr(axes, "pymetrix_panel", 0)
        if page.picking["kind"] == "change":
            lines = [(l, _nearest(l.axes, l.get_xdata(orig=False), l.get_ydata(orig=False), event.x, event.y))
                     for a in canvas.figure.axes if getattr(a, "pymetrix_panel", None) == panel for l in a.get_lines() if _name(l)]
            lines = [(l, n) for l, n in lines if n is not None]
            first = page.picking["points"][0][3] if page.picking["points"] else None
            if first is not None:  # the second point is on the same series
                lines = [(l, n) for l, n in lines if l.get_label() == first] or lines
            if not lines:
                return None
            line, (_, x, y) = min(lines, key=lambda item: item[1][0])
            return panel, _period(line.axes, x), y, line.get_label()
        xdata, ydata = axes.transData.inverted().transform((event.x, event.y))
        return panel, _period(axes, float(xdata)), float(f"{ydata:.6g}"), ""  # the nearest period on a dated axis

    def rubber_band(event: Any) -> None:
        """While drawing: an arrow from the first point to the mouse, on the canvas only (never recorded)."""
        picking = page.picking
        if picking is None or picking["press"] is None:
            return
        axes, start = picking["axes"], picking["start"]
        end = tuple(axes.transData.inverted().transform((event.x, event.y)))
        if picking["rubber"] is None:
            from .chart_edit import GID

            style = "<->" if picking["kind"] == "change" else "->"
            picking["rubber"] = axes.annotate("", xy=end, xytext=start, annotation_clip=False, zorder=1001,
                                              arrowprops={"arrowstyle": style, "color": theme.CHART["arrow"], "linewidth": 1.5})
            picking["rubber"].set_gid(GID)
            picking["rubber"].set_in_layout(False)
        else:
            picking["rubber"].xy = end
        canvas.draw_idle()

    def finish(event: Any) -> None:
        point = picked_point(event, page.picking["axes"])
        if point is None:
            return
        points = page.picking["points"]
        points.append(point)
        kind = page.picking["kind"]
        stop_picking()
        (panel, x0, y0, series), (_, x1, y1, _) = points
        on_pick(kind, panel, (x0, y0), (x1, y1), series)

    def pick_press(event: Any) -> None:
        """Draw an arrow or Mark a change: press where it starts and drag to where it ends, or click twice."""
        if page.picking is None or event.button != 1 or getattr(page.toolbar, "mode", ""):
            return
        if page.picking["press"] is not None:  # the second click
            finish(event)
            return
        point = picked_point(event)
        if point is None:
            return
        page.picking.update(points=[point], press=(event.x, event.y), axes=event.inaxes,
                            start=(float(event.xdata), float(event.ydata)))
        page.readout.setText("drag to where it ends, or click there (Esc: stop)")

    def pick_release(event: Any) -> None:
        picking = page.picking
        if picking is None or picking["press"] is None or event.button != 1:
            return
        px, py = picking["press"]
        if abs(event.x - px) + abs(event.y - py) > 5:  # dragged: done; a click waits for the second
            finish(event)

    def pick_key(event: Any) -> None:
        if event.key == "escape" and page.picking is not None:
            stop_picking()

    page.arrow_action = None
    if on_pick is not None:
        canvas.mpl_connect("button_press_event", pick_press)
        canvas.mpl_connect("motion_notify_event", rubber_band)
        canvas.mpl_connect("button_release_event", pick_release)
        canvas.mpl_connect("key_press_event", pick_key)
        arrow_action = page.toolbar.addAction("Arrow")
        arrow_action.setCheckable(True)
        arrow_action.setToolTip("draw an arrow: press where it starts and drag to where it points. Then drag it, or "
                                "its ends, to adjust it; double-click it to format it; Delete removes it")
        arrow_action.toggled.connect(lambda on: start_picking("arrow", ARROW_SAY) if on else stop_picking())
        page.arrow_action = arrow_action
    page.start_picking = start_picking
    page.stop_picking = stop_picking
    for label, handler, tip in (("Copy image", copy_image, "copy the figure, to paste into Word or PowerPoint"),
                                ("Save…", save, "save as PNG, SVG or PDF"),
                                ("Copy data",  # \n everywhere, as tables' and matrices' copies (pandas would use \r\n on Windows)
                                 lambda: Qt.QApplication.clipboard().setText(figure.data().to_csv(sep="\t", lineterminator="\n")),
                                 "copy the series drawn, tab-separated, for Excel"),
                                ("Copy plotting code", lambda: Qt.QApplication.clipboard().setText(figure.matplotlib_code()),
                                 "copy a standalone matplotlib script that draws this figure from its numbers")):
        button = Qt.QPushButton(label)
        button.setToolTip(tip)
        button.clicked.connect(lambda _=False, h=handler: h())
        bar.addWidget(button)
    if on_report is not None:
        report = Qt.QPushButton("Add to report")
        report.setToolTip("put this figure in the report being put together (the Report panel)")
        report.clicked.connect(lambda: on_report())
        bar.addWidget(report)
        page.add_to_report = report
    bar.addStretch()
    layout.addWidget(bar)
    page.actions_row = bar
    page.canvas = canvas

    def context_menu(x: float, y: float) -> Any:
        """The right-click menu for the point (x, y) in matplotlib's display coordinates: what is under it first."""
        from matplotlib.backend_bases import MouseEvent

        from .chart_click import identify

        event = MouseEvent("button_press_event", canvas, x, y, button=3)
        menu = Qt.QMenu(page)
        menu.setToolTipsVisible(True)
        axes = event.inaxes
        panel = getattr(axes, "pymetrix_panel", None) if axes is not None else None
        found = page.editor.artist_at(event) if page.editor is not None and page.picking is None else None
        if found is not None:
            step, call, _ = found
            page.editor.select(step)
            words = {"annotate": "text", "text": "s"}.get(call.kind)
            menu.addAction("Format…", lambda: page.format_item(step))
            if words is not None:
                menu.addAction("Edit the words…", lambda: page.editor.on_text(step, words, call.props.get(words, "")))
            menu.addAction("Bring to front", lambda: on_restyle(step, {"zorder": 5}))
            menu.addAction("Send to back", lambda: on_restyle(step, {"zorder": 1}))
            if on_format is not None:
                menu.addAction("Edit as code (every property)…", lambda: on_format(step))
            menu.addSeparator()
            menu.addAction("Delete", lambda: (page.editor.select(None), on_remove(step)))
            menu.addSeparator()
        elif axes is not None and getattr(axes, "name", "") != "3d" and on_style is not None:
            def on_it(artist: Any, found: dict[str, Any]) -> bool:  # near a data point, or on the line between two
                try:
                    return found.get("distance", 99) <= 8 or bool(artist.contains(event)[0])
                except Exception:  # noqa: BLE001 - an artist matplotlib cannot hit-test is not under the mouse
                    return False

            near = [f for a in canvas.figure.axes if getattr(a, "pymetrix_panel", None) == panel
                    for artist in [*a.get_lines(), *a.collections, *a.patches]
                    for f in [identify(artist, x, y)] if f is not None and on_it(artist, f)]
            if near:
                series = min(near, key=lambda f: f["distance"])["text"].split("\n")[0]
                menu.addAction(f"Format series “{series}”…", lambda: on_style(series, panel))
                menu.addSeparator()
        if axes is not None and getattr(axes, "name", "") != "3d":
            if on_style is not None:
                menu.addAction("Format axes…", lambda: on_style("", panel)).setToolTip("limits, scale, labels, ticks, grid and legend")
            if on_pick is not None:
                menu.addAction("Draw an arrow", lambda: start_picking("arrow", ARROW_SAY))
            if on_place is not None and axes.get_xlim():
                import pandas as pd
                from matplotlib import dates

                freq = getattr(axes, "pymetrix_freq", None)
                xdata, ydata = axes.transData.inverted().transform((x, y))
                where: Any = (str(pd.Period(dates.num2date(xdata).replace(tzinfo=None), freq=freq)) if freq is not None
                              else f"{xdata:.4g}")
                menu.addAction("Add a note here…", lambda: on_place(panel, where, float(ydata)))
            menu.addSeparator()
        if on_edit is not None:
            menu.addAction("Titles, shading and marks…", lambda: on_edit("options"))
        menu.addAction("Copy image", copy_image)
        menu.addAction("Save…", save)
        return menu

    def context_requested(pos: Any) -> None:
        if getattr(page.toolbar, "mode", ""):  # zoom or pan: the right button is theirs
            return
        x, y = canvas.mouseEventCoords(pos)
        page.identify.dismiss()
        context_menu(x, y).popup(canvas.mapToGlobal(pos))

    page.context_menu = context_menu
    from PySide6.QtCore import Qt as QtCoreNS

    canvas.setContextMenuPolicy(QtCoreNS.ContextMenuPolicy.CustomContextMenu)
    canvas.customContextMenuRequested.connect(context_requested)
    return page


def trust_dialog(parent: Any, name: str, info: Any) -> Any:
    """Ask before running a project's typed code: show the code, and offer to run it or open without it.
    ``dialog.choice`` is True (run and trust), False (open without the code) or None (cancel)."""
    Qt = _qt()
    dialog = Qt.QDialog(parent)
    dialog.setWindowTitle(f"Open {name}")
    dialog.choice = None
    layout = Qt.QVBoxLayout(dialog)
    metrics.set_margins(layout, "dialog_margin")
    count = len(info.steps)
    text = Qt.QLabel(
        f"<b>{name}</b> contains {count} step{'s' * (count != 1)} of Python code typed in the Code or Immediate "
        "windows. Python code can do anything a program on this computer can, so it runs only if you trust "
        "the project. Read it below.<br><br>Opening without the code restores the data and the dialog steps; "
        "the code is kept, marked as not run, and steps that need its results are not run either.")
    text.setWordWrap(True)
    layout.addWidget(text)
    code = Qt.QPlainTextEdit()
    code.setReadOnly(True)
    code.setPlainText("\n\n".join(f"# step {i}\n{step}" for i, step in enumerate(info.steps, 1)))
    code.setMinimumSize(560, 260)
    layout.addWidget(code, 1)
    buttons = Qt.QDialogButtonBox()
    run = buttons.addButton("Run the code and trust this project", Qt.QDialogButtonBox.ButtonRole.AcceptRole)
    without = buttons.addButton("Open without running the code", Qt.QDialogButtonBox.ButtonRole.ActionRole)
    cancel = buttons.addButton(Qt.QDialogButtonBox.StandardButton.Cancel)
    without.setDefault(True)

    def choose(value: Any) -> None:
        dialog.choice = value
        dialog.accept() if value is not None else dialog.reject()

    run.clicked.connect(lambda: choose(True))
    without.clicked.connect(lambda: choose(False))
    cancel.clicked.connect(lambda: choose(None))
    layout.addWidget(buttons)
    dialog.code, dialog.buttons = code, {"run": run, "without": without, "cancel": cancel}
    return dialog


def ask_trust(parent: Any, name: str, info: Any) -> bool | None:
    dialog = trust_dialog(parent, name, info)
    dialog.exec()
    return dialog.choice


def _is_result(obj: Any) -> bool:
    """A result object (it gets a Python access view), not data, a table, a figure or an array."""
    import numpy as np
    import pandas as pd

    from ..data import Dataset
    from ..figures import Figure
    from ..output import Table

    return not isinstance(obj, (Dataset, Table, Figure, pd.DataFrame, pd.Series, np.ndarray, str, int, float, list, tuple, dict))


def _selected_text(grid: Any, sep: str = "\t") -> str:
    """The selected cells as rows of text (with their row and column names when whole rows or columns are chosen)."""
    cells = sorted((i.row(), i.column()) for i in grid.selectedIndexes())
    if not cells:
        return ""
    rows = sorted({r for r, _ in cells})
    cols = sorted({c for _, c in cells})
    chosen = set(cells)
    header = [""] + [grid.horizontalHeaderItem(c).text() if grid.horizontalHeaderItem(c) else "" for c in cols]
    lines = [sep.join(header)]
    for r in rows:
        name = grid.verticalHeaderItem(r).text() if grid.verticalHeaderItem(r) else str(r + 1)
        lines.append(sep.join([name] + [_copy_text(grid.item(r, c)) if (r, c) in chosen else "" for c in cols]))
    return "\n".join(lines)


def _copy_text(item: Any) -> str:
    """A cell as copied: the number it holds at full precision (not the four decimals, or "<0.0001", shown)."""
    if item is None:
        return ""
    from PySide6.QtCore import Qt as QtNS

    value = item.data(QtNS.ItemDataRole.UserRole)
    return repr(value) if isinstance(value, float) else str(value) if isinstance(value, int) else item.text()


def _table_menu(grid: Any, table: Any, fields: bool = False, report: Any = None, heatmap: Any = None) -> None:
    """Right-click on a table: copy the selected cells or the whole table, for Excel, as Markdown or as CSV; on the
    Python access view, the Python path of the row (e.g. fit.stats["aic"]); sort the rows shown by the column clicked
    (the view only: the result is unchanged), back to their order; and, given them, ``heatmap()`` (Show as a heatmap,
    a recorded step) and ``report()`` (Add to report). ``grid.menu_at(pos)`` opens it; ``grid.last_menu`` lists it."""
    Qt = _qt()
    from PySide6.QtCore import Qt as QtNS

    grid.setContextMenuPolicy(QtNS.ContextMenuPolicy.CustomContextMenu)
    grid.setSelectionMode(Qt.QAbstractItemView.SelectionMode.ExtendedSelection)

    def put(text: str) -> None:
        Qt.QApplication.clipboard().setText(text)

    def menu_at(pos: Any) -> None:
        menu = Qt.QMenu(grid)
        if grid.selectedIndexes():
            menu.addAction("Copy selected cells").triggered.connect(lambda: put(_selected_text(grid)))
        if fields:
            row = grid.indexAt(pos).row()
            header = grid.verticalHeaderItem(row) if row >= 0 else None
            if header is not None:
                menu.addAction(f"Copy Python: {header.text()}").triggered.connect(lambda: put(header.text()))
        menu.addSeparator()
        menu.addAction("Copy table, for Excel").triggered.connect(lambda: put(table.to_csv(sep="\t")))
        menu.addAction("Copy table as Markdown").triggered.connect(lambda: put(table.to_markdown()))
        menu.addAction("Copy table as CSV").triggered.connect(lambda: put(table.to_csv()))
        column = grid.indexAt(pos).column()
        if column >= 0 and grid.rowCount() > 1:
            menu.addSeparator()
            heading = str(frame.columns[column]) if column < len(frame.columns) else str(column)
            menu.addAction(f"Sort by “{heading}”, smallest first").triggered.connect(lambda: sort_by(column, True))
            menu.addAction(f"Sort by “{heading}”, largest first").triggered.connect(lambda: sort_by(column, False))
            if grid.sorted_by is not None:
                menu.addAction("Back to the table's order").triggered.connect(lambda: sort_by(None, True))
        if heatmap is not None or report is not None:
            menu.addSeparator()
        if heatmap is not None:
            menu.addAction("Show as a heatmap").triggered.connect(lambda: heatmap())
        if report is not None:
            menu.addAction("Add to report").triggered.connect(lambda: report())
        grid.last_menu = [a.text() for a in menu.actions() if a.text()]
        grid.last_menu_actions = {a.text(): a for a in menu.actions() if a.text()}
        menu.popup(grid.viewport().mapToGlobal(pos))

    import pandas as pd

    frame = table.to_frame()
    grid.sorted_by = None

    def sort_by(column: int | None, ascending: bool) -> None:
        """The rows shown sorted by a column (numbers as numbers, missing values last), or back in the table's order."""
        if column is None:
            shown = frame
        else:
            key = frame.columns[column]
            numbers = pd.to_numeric(frame[key], errors="coerce")
            order = (numbers if numbers.notna().any() else frame[key].astype(str)).sort_values(ascending=ascending,
                                                                                                na_position="last").index
            shown = frame.loc[order]
        if table.index is not None or column is not None:
            grid.setVerticalHeaderLabels([str(i) for i in shown.index])
        _fill_grid(Qt, grid, shown, table)
        grid.sorted_by = None if column is None else (column, ascending)

    grid.sort_by = sort_by
    grid.customContextMenuRequested.connect(menu_at)
    grid.menu_at = menu_at


def _scoped_dataset(data: Any, names: Any) -> Any:
    """The dataset with only these series, to show (not to compute with: derived series keep their recipes)."""
    from ..data import Dataset

    keep = set(names)
    cols = [n for n in data.names if n in keep]
    flags = data.flags[[c for c in cols if c in data.flags.columns]] if data.flags is not None else None
    return Dataset(data.frame[cols], data.axes, {n: data.variables[n] for n in cols}, flags, data.meta)


MANY_SERIES = 300  # a dataset this large suggests a working set


def series_browser(table: Any, actions: dict[str, Any] | None = None) -> Any:
    """A dataset's series as a table to search, sort and act on: its metadata, sample and gaps (the Variables view),
    a filter over every column, sorting by any column, and, for the series picked, Plot…, Edit in spreadsheet,
    New selection…, Generate for each… and Use as working set (``actions``: name -> callable taking the series' names)."""
    Qt = _qt()
    from PySide6.QtCore import QSortFilterProxyModel
    from PySide6.QtCore import Qt as QtNS
    from PySide6.QtGui import QStandardItem, QStandardItemModel

    from .fit import ActionRow, scrolled_page

    frame = table.to_frame()
    page, body = scrolled_page()
    layout = Qt.QVBoxLayout(body)
    top = Qt.QHBoxLayout()
    search = Qt.QLineEdit()
    search.setPlaceholderText("Filter series by name, label, units, source…")
    search.setClearButtonEnabled(True)
    count = Qt.QLabel()
    count.setStyleSheet(theme.muted_css())
    top.addWidget(search, 1)
    top.addWidget(count)
    layout.addLayout(top)
    columns = ["series", *[str(c) for c in frame.columns]]
    model = QStandardItemModel(len(frame), len(columns))
    # headers in words ("Source key", not source_key), the metadata field in each one's tooltip
    model.setHorizontalHeaderLabels([c[:1].upper() + c[1:].replace("_", " ") for c in columns])
    for c, key in enumerate(columns):
        model.horizontalHeaderItem(c).setToolTip(f"{key}" if c else "the series' name")
    for r, (name, row) in enumerate(frame.iterrows()):
        model.setItem(r, 0, QStandardItem(str(name)))
        for c, value in enumerate(row, start=1):
            item = QStandardItem()
            if value is None or (isinstance(value, float) and value != value):
                item.setText("")
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                item.setData(value, QtNS.ItemDataRole.DisplayRole)  # sorts as a number
            else:
                item.setText(str(value))
            model.setItem(r, c, item)
    proxy = QSortFilterProxyModel(page)
    proxy.setSourceModel(model)
    proxy.setFilterKeyColumn(-1)  # every column
    proxy.setFilterCaseSensitivity(QtNS.CaseSensitivity.CaseInsensitive)
    view = Qt.QTableView()
    view.setModel(proxy)
    view.setSortingEnabled(True)
    view.setSelectionBehavior(Qt.QAbstractItemView.SelectionBehavior.SelectRows)
    view.setSelectionMode(Qt.QAbstractItemView.SelectionMode.ExtendedSelection)
    view.setEditTriggers(Qt.QAbstractItemView.EditTrigger.NoEditTriggers)
    view.verticalHeader().setVisible(False)
    view.horizontalHeader().setStretchLastSection(True)
    layout.addWidget(view, 1)

    def picked() -> list[str]:
        rows = sorted({i.row() for i in view.selectionModel().selectedRows()})
        return [proxy.index(r, 0).data() for r in rows]

    def update() -> None:
        n = len(picked())
        count.setText(f"{proxy.rowCount():,} of {model.rowCount():,} series"
                      + (f", {n} picked" if n else (": select series to act on them" if buttons else "")))
        for button in buttons.values():
            button.setEnabled(n > 0)
            tip = button.property("tip") or button.toolTip()
            button.setProperty("tip", tip)
            button.setToolTip(tip if n else f"{tip} (select series in the table first)")

    bar = ActionRow()  # what does not fit goes in its » menu
    buttons: dict[str, Any] = {}
    labels = {"plot": ("Plot…", "plot the series picked"), "sheet": ("Edit in spreadsheet", "edit the series picked, only those shown"),
              "select": ("New selection…", "name the series picked as a selection"),
              "generate": ("Generate for each…", "make a new series from each series picked, by a name and a formula template"),
              "working_set": ("Use as working set", "show only the series picked in dialogs, the spreadsheet and this window")}
    menu = Qt.QMenu(view)
    for key, (label, tip) in labels.items():
        if not actions or key not in actions:
            continue
        button = Qt.QPushButton(label)
        button.setToolTip(tip)
        button.clicked.connect(lambda _=False, k=key: actions[k](picked()))
        bar.addWidget(button)
        buttons[key] = button
        menu.addAction(label).triggered.connect(lambda _=False, k=key: actions[k](picked()))
    bar.addStretch()
    layout.addWidget(bar)
    page.actions_row = bar
    if buttons:
        view.setContextMenuPolicy(QtNS.ContextMenuPolicy.CustomContextMenu)
        view.customContextMenuRequested.connect(lambda pos: menu.popup(view.viewport().mapToGlobal(pos)) if picked() else None)
    search.textChanged.connect(lambda text: (proxy.setFilterFixedString(text), update()))
    view.selectionModel().selectionChanged.connect(lambda *_: update())
    update()
    page.search, page.view, page.proxy, page.picked, page.buttons, page.count = search, view, proxy, picked, buttons, count
    return page


def object_view(obj: Any, parent: Any = None, name: str = "", code: str = "", edit_figure: Any = None, place_note: Any = None,
                heatmap: Any = None, scope: str = "", series_actions: dict[str, Any] | None = None,
                add_to_report: Any = None, next_steps: list[tuple[str, Any]] | None = None, matrix_menu: Any = None,
                figure_edits: Any = None) -> Any:
    """Tabs of tables for an object (Summary, Coefficients, Statistics, ...), each with Copy, Copy as and Export.
    A result also gets a Python access view: its named fields, the Python for each, and the command that made it.
    ``edit_figure(tab, figure, kind)``, when given, puts an Edit… button on each figure; ``heatmap(tab)`` a Heatmap
    button on each table of numbers. ``scope`` says, for a dataset shown in part, which part ("24 of 2,140 series
    (working set: gdp_*)"). A dataset's Variables tab is a series browser (``series_browser``), with
    ``series_actions`` for the series picked in it. ``add_to_report(tab)``, when given, puts Add to report under each
    figure and table (and, with ``tab`` None, in the corner, for the result's summary and graph). ``next_steps``,
    (label, callable) pairs, make the corner's Next menu: the next stage, with this result filled in. ``matrix_menu(parent)``,
    for a matrix or a decomposition, gives the corner's Matrix menu (Transpose, Inverse, Decompose, Copy as LaTeX, ...).
    ``figure_edits(tab, figure, action, *args)``, when given, takes a figure's edits made on it: ``"pick"`` (kind,
    panel, first, second, series) after Draw an arrow or Mark a change, ``"restyle"`` (step, props) after a drag or
    an edit of words, ``"remove"`` (step) after Delete."""
    import pandas as pd

    Qt = _qt()
    from ..views import fields, view_sections as sections

    from ..figures import Figure

    from .fit import ActionRow, scrolled_page

    from .view_switcher import ViewSwitcher

    tabs = ViewSwitcher(parent, kind=type(obj).__name__)  # the views listed down the side, grouped: all in view
    tabs.tables = {}
    tabs.figures = {}
    parts = list(sections(obj))
    from ..data import Dataset

    if isinstance(obj, Dataset):  # its name, dimensions, observations and sample, first
        from ..views import dataset_overview, dataset_status

        parts.insert(0, ("Overview", dataset_overview(obj, name)))
        status = dataset_status(obj)
        if scope:
            status = f"{scope} · {status}"
        elif len(obj.names) > MANY_SERIES:
            status = f"{status} · many series: right-click it in the workspace for a working set"
        header = Qt.QLabel(" " + status + " ")  # always in view, whichever tab is open
        header.setToolTip("series, frequency, periods, rows, active sample, derived and sourced series, missing values"
                          + ("; the series shown are the dataset's working set (right-click it in the workspace to change it)" if scope else ""))

        def next_menu() -> Any:
            button = Qt.QToolButton()
            button.setText("Next ▸")
            button.setToolTip("the next step, with this filled in (in that stage's workspace)")
            button.setPopupMode(Qt.QToolButton.ToolButtonPopupMode.InstantPopup)
            menu = Qt.QMenu(button)
            for label, run in next_steps or ():
                if label is None:  # the dialogs, then the task pages
                    menu.addSeparator()
                    continue
                menu.addAction(label).triggered.connect(lambda _=False, r=run: r())
            button.setMenu(menu)
            tabs.next_menu = menu
            return button

        corner = ActionRow()  # beside the tabs, never over them: what does not fit goes in its » menu
        corner.addWidget(header)
        if next_steps:  # the header, and the next steps for the data
            corner.addWidget(next_menu())
        tabs.setCornerWidget(corner)
        tabs.corner = corner
        tabs.header = header
    elif code or _is_result(obj):  # one click to the result as text, and to the step that made it
        corner = row = ActionRow()  # beside the tabs, never over them: what does not fit goes in its » menu
        metrics.set_spacing(row, "tight")
        summary = Qt.QToolButton()
        summary.setText("Copy summary")
        summary.setToolTip("the result as text, as print(result.summary()) shows it: for an email or notes")
        from ..views import summary_text

        summary.clicked.connect(lambda: Qt.QApplication.clipboard().setText(summary_text(obj)))
        row.addWidget(summary)
        tabs.copy_summary = summary
        if code:
            made_by = Qt.QToolButton()
            made_by.setText("Copy the step's code")
            made_by.setToolTip(f"the Python step that made this result:\n{code}")
            made_by.clicked.connect(lambda: Qt.QApplication.clipboard().setText(code))
            row.addWidget(made_by)
            tabs.made_by = made_by

        def next_menu() -> Any:
            button = Qt.QToolButton()
            button.setText("Next ▸")
            button.setToolTip("the next step, with this filled in (in that stage's workspace)")
            button.setPopupMode(Qt.QToolButton.ToolButtonPopupMode.InstantPopup)
            menu = Qt.QMenu(button)
            for label, run in next_steps or ():
                if label is None:  # the dialogs, then the task pages
                    menu.addSeparator()
                    continue
                menu.addAction(label).triggered.connect(lambda _=False, r=run: r())
            button.setMenu(menu)
            tabs.next_menu = menu
            return button

        if next_steps:
            row.addWidget(next_menu())
        if matrix_menu is not None:
            row.addWidget(_matrix_button(tabs, matrix_menu))
        if add_to_report is not None:
            whole = Qt.QToolButton()
            whole.setText("Add result to report")  # the whole result; under each table or figure: that view
            whole.setToolTip("put this result in the report being put together: its summary and graph")
            whole.clicked.connect(lambda: add_to_report(None))
            row.addWidget(whole)
            tabs.add_result_to_report = whole
        tabs.setCornerWidget(corner)
        tabs.corner = corner
    elif matrix_menu is not None:
        corner = ActionRow()
        corner.addWidget(_matrix_button(tabs, matrix_menu))
        tabs.setCornerWidget(corner)
        tabs.corner = corner
    if _is_result(obj):
        parts.append((FIELDS_VIEW, fields(obj, name or "result", code)))
    for title, table in parts:
        if isinstance(obj, Dataset) and title == "Variables":  # the series: to search, sort and act on
            browser = series_browser(table, series_actions)
            tabs.addTab(browser, title)
            tabs.browser = browser
            tabs.tables[title] = table.to_frame()
            continue
        if isinstance(table, Figure):
            on_edit = None if edit_figure is None else (lambda kind="options", t=title, f=table: edit_figure(t, f, kind))
            on_place = None if place_note is None else (lambda panel, x, y, t=title, f=table: place_note(t, f, panel, x, y))
            on_report = None if add_to_report is None else (lambda t=title: add_to_report(t))
            edits = {} if figure_edits is None else {
                "on_pick": lambda *a, t=title, f=table: figure_edits(t, f, "pick", *a),
                "on_restyle": lambda *a, t=title, f=table: figure_edits(t, f, "restyle", *a),
                "on_remove": lambda *a, t=title, f=table: figure_edits(t, f, "remove", *a),
                "on_format": lambda *a, t=title, f=table: figure_edits(t, f, "format", *a),
                "on_style": lambda *a, t=title, f=table: figure_edits(t, f, "style", *a)}
            tabs.addTab(figure_view(table, title, on_edit=on_edit, on_place=on_place, on_report=on_report, **edits), title, figure=True)
            tabs.figures[title] = table
            continue
        page, body = scrolled_page()  # a window shorter than the title, table, notes and buttons scrolls
        layout = Qt.QVBoxLayout(body)
        frame = table.to_frame()
        grid = Qt.QTableWidget(len(frame), len(frame.columns))
        grid.setHorizontalHeaderLabels([str(c) for c in frame.columns])
        if table.index is not None:
            grid.setVerticalHeaderLabels([str(i) for i in frame.index])
        grid.setEditTriggers(Qt.QAbstractItemView.EditTrigger.NoEditTriggers)
        _fill_grid(Qt, grid, frame, table)
        numeric = frame.apply(pd.to_numeric, errors="coerce").notna().to_numpy().sum() if len(frame) else 0
        _table_menu(grid, table, title == FIELDS_VIEW,
                    report=(lambda t=title: add_to_report(t)) if add_to_report is not None and title != FIELDS_VIEW else None,
                    heatmap=(lambda t=title: heatmap(t)) if heatmap is not None and title != FIELDS_VIEW and len(frame) > 1
                    and numeric >= 4 else None)
        if table.title:
            layout.addWidget(Qt.QLabel(f"<b>{table.title}</b>"))
        layout.addWidget(grid, 1)
        if not len(frame):  # an empty state, not a blank grid
            nothing = Qt.QLabel("Nothing to show: this table has no rows.")
            nothing.setStyleSheet(theme.muted_css())
            layout.addWidget(nothing)
            grid.setVisible(False)
        if table.notes:  # the statistics grouped in columns, then the notes that are sentences
            from .stats_panel import StatsPanel

            layout.addWidget(StatsPanel(table.notes))
        layout.addStretch(0)  # a short table and its notes stay together at the top; a long one keeps the room
        bar = ActionRow()  # what does not fit goes in its » menu
        copy = Qt.QPushButton("Copy")
        export = Qt.QPushButton("Export…")
        copy.setToolTip("copy the table with its headers, tab-separated, for pasting into Excel")
        copy.clicked.connect(lambda _=False, t=table: Qt.QApplication.clipboard().setText(t.to_csv(sep="\t")))
        copy_as = Qt.QPushButton("Copy as")
        copy_as.setToolTip("copy the values only, as Markdown, or the Python that gives this table")
        menu = Qt.QMenu(copy_as)
        python = (f"from pymetrix.views import view\nview({name}, {title!r}).to_frame()" if name and title != FIELDS_VIEW
                  else f"from pymetrix.views import fields\nfields({name}, {name!r}).to_frame()" if name else "")
        for label, text in (("Values only, for Excel", lambda t=table: t.to_csv(sep="\t", header=False)),
                            ("Markdown", lambda t=table: t.to_markdown()),
                            ("Python", lambda p=python: p)):
            action = menu.addAction(label)
            action.triggered.connect(lambda _=False, f=text: Qt.QApplication.clipboard().setText(f()))
            action.setEnabled(label != "Python" or bool(python))
        copy_as.setMenu(menu)
        tabs.copy_texts = getattr(tabs, "copy_texts", {})
        tabs.copy_texts[title] = {"excel": table.to_csv(sep="\t"), "values": table.to_csv(sep="\t", header=False),
                                  "markdown": table.to_markdown(), "python": python}

        def save(_: bool = False, f: Any = frame, name: str = title) -> None:
            path, _ = Qt.QFileDialog.getSaveFileName(tabs, "Export table", f"{name}.csv", "CSV (*.csv);;Excel (*.xlsx)")
            if path:
                f.to_excel(path) if path.lower().endswith(".xlsx") else f.to_csv(path)

        export.clicked.connect(save)
        bar.addWidget(copy)
        bar.addWidget(copy_as)
        bar.addWidget(export)
        if add_to_report is not None and title != FIELDS_VIEW:
            to_report = Qt.QPushButton("Add to report")
            to_report.setToolTip("put this table in the report being put together (the Report panel)")
            to_report.clicked.connect(lambda _=False, t=title: add_to_report(t))
            bar.addWidget(to_report)
        if heatmap is not None and title != FIELDS_VIEW and len(frame) > 1 and numeric >= 4:  # a table of numbers to colour
            colour = Qt.QPushButton("Heatmap")
            colour.setToolTip("show the table's numbers as coloured cells (recorded as a step): on one scale, or each column on "
                              "its own where the columns are in different units")
            colour.clicked.connect(lambda _=False, t=title: heatmap(t))
            bar.addWidget(colour)
            tabs.heatmap_buttons = getattr(tabs, "heatmap_buttons", {})
            tabs.heatmap_buttons[title] = colour
        bar.addStretch()
        layout.addWidget(bar)
        page.actions_row = bar
        tabs.addTab(page, title)
        tabs.tables[title] = frame
        tabs.grids = getattr(tabs, "grids", {})
        tabs.grids[title] = grid

    tabs.restore()  # the view last picked for this kind of result

    def show_at(factor: float) -> None:
        from .zoom import scale_font

        scale_font(tabs, factor)
        for grid in getattr(tabs, "grids", {}).values():  # columns and rows follow the text
            _size_columns(grid)
            _fit_height(grid)
        for child in tabs.findChildren(Qt.QWidget):
            if hasattr(child, "figure") and hasattr(child, "draw_idle"):  # a matplotlib canvas
                fig = child.figure
                if not hasattr(child, "base_dpi"):
                    child.base_dpi = fig.dpi
                fig.set_dpi(child.base_dpi * factor)
                ratio = getattr(child, "device_pixel_ratio", 1) or 1  # the canvas' pixels: logical ones times the ratio
                fig.set_size_inches(max(child.width(), 1) * ratio / fig.dpi, max(child.height(), 1) * ratio / fig.dpi, forward=False)
                child.draw_idle()

    from .zoom import make_zoomable

    return make_zoomable(tabs, show_at)


def _matrix_button(tabs: Any, matrix_menu: Any) -> Any:
    """The corner's Matrix ▸ button: steps that make a new matrix from this one, and ways to copy or export it."""
    Qt = _qt()
    button = Qt.QToolButton()
    button.setText("Matrix ▸")
    button.setToolTip("a new matrix from this one (each a step), or this one as LaTeX, CSV or an Excel file")
    button.setPopupMode(Qt.QToolButton.ToolButtonPopupMode.InstantPopup)
    menu = matrix_menu(button)
    button.setMenu(menu)
    tabs.matrix_menu = menu
    return button


def main_window(session: Session) -> Any:
    """Objects and Script are dock panels; each object opens in its own titled window."""
    Qt = _qt()
    from PySide6.QtCore import QSettings
    from PySide6.QtCore import Qt as QtNS

    from ..commands import load_project, save_project

    settings = QSettings(QSettings.defaultFormat(), QSettings.Scope.UserScope, "pymetrix", "desktop")  # tests point the default elsewhere
    from . import collector, text_size

    collector.install(Qt.QApplication.instance())  # Python's cyclic collector on this (the GUI) thread only
    text_size.apply(text_size.saved())  # View → Text size, as last chosen: before any widget is made
    state = {"session": session}

    from PySide6.QtCore import QEvent, QTimer

    layout_fit: dict[str, bool] = {"pending": False}

    def refit() -> None:
        layout_fit["queued"] = False
        if layout_fit["pending"]:
            fit_default_layout()
  # the panels are in a default arrangement: fit them to the window as it resizes

    class MainWindow(Qt.QMainWindow):
        def resizeEvent(self, event: Any) -> None:  # noqa: N802
            super().resizeEvent(event)
            if layout_fit["pending"] and not layout_fit.get("queued"):  # once the layout has taken the new size
                layout_fit["queued"] = True
                QTimer.singleShot(0, refit)

        def event(self, event: Any) -> bool:
            if event.type() == QEvent.Type.MouseButtonPress:  # a panel's edge taken to drag (QMainWindow.event handles
                layout_fit["pending"] = False  # it, not mousePressEvent): the user's arrangement from now on
            return super().event(event)

        def dragEnterEvent(self, event: Any) -> None:  # noqa: N802
            if dropped_files(event.mimeData()):
                event.acceptProposedAction()

        def dropEvent(self, event: Any) -> None:  # noqa: N802
            files = dropped_files(event.mimeData())
            if files:
                event.acceptProposedAction()
                open_dropped(files, dropped=sum(u.isLocalFile() for u in event.mimeData().urls()))

        def closeEvent(self, event: Any) -> None:
            if not all(e.confirm_close() for e in [*editors(), *sheets.values()]) or not confirm_discard():
                event.ignore()
                return
            settings.setValue("geometry", self.saveGeometry())
            settings.setValue("state", self.saveState())
            stop_autosave()  # closed on purpose: no copy to recover
            if _PROGRESS.get("window") is self:  # its own progress bar only: a newer window may have registered since
                _PROGRESS.clear()
            _MENUS[:] = [m for m in _MENUS if not _belongs_to(m, self)]  # its own menus; a newer window keeps its own
            from ..plugins import set_mode

            set_mode("script")  # code run after the window closes treats new plugins as scripts do
            if getattr(self, "workspace", None):  # the workspace's own arrangement too
                settings.setValue(f"workspace/{self.workspace}/state", self.saveState())
            if immediate is not None:
                immediate.shutdown()
            app_interface._close()  # its plugins' panels stop following the history (code run in a later window)
            _show_closed(show_token)  # show() to the newest window still open, else back to printing
            super().closeEvent(event)

    def dropped_files(mime: Any) -> list[str]:
        """The files dropped that the app opens: projects and data files."""
        if not mime.hasUrls():
            return []
        paths = [u.toLocalFile() for u in mime.urls() if u.isLocalFile()]
        return [p for p in paths if p.lower().endswith((".pymx", *DATA_FILES))]

    def open_dropped(paths: list[str], dropped: int | None = None) -> None:
        """A dropped project opens; a dropped data file opens Load data with it filled in, named after the file (a
        name not yet taken). One file opens at a time: the status bar says which, and how many others were left."""
        from ..commands import LoadData

        path = next((p for p in paths if p.lower().endswith(".pymx")), paths[0])
        others = max(len(paths), dropped or 0) - 1
        if others:
            window.statusBar().showMessage(f"Opening {Path(path).name}; {others} other file{'s' if others != 1 else ''} "
                                           "not opened (drop one file at a time, or a project)", 15000)
        if path.lower().endswith(".pymx"):
            open_project(path)
            return
        stem = "".join(c if c.isalnum() else "_" for c in Path(path).stem).strip("_") or "data"
        base = stem if not stem[0].isdigit() else f"data_{stem}"
        taken = current().objects
        name = next(n for n in (base, *(f"{base}_{i}" for i in range(2, 10_000))) if n not in taken)
        run_dialog(LoadData, initial=LoadData(path, name=name))

    window = MainWindow()
    window.setAcceptDrops(True)
    from pymetrix_app import ICON

    if Qt.QApplication.instance().windowIcon().isNull() and ICON.is_file():  # pymetrix-gui sets it (identify); a window
        from PySide6.QtGui import QIcon  # built otherwise (python -m, a notebook, a test) shows the app's, not Qt's

        Qt.QApplication.instance().setWindowIcon(QIcon(str(ICON)))
    Dock = _dock_class(Qt)  # panels: their default sizes (default_dock_size)
    mdi = Qt.QMdiArea()
    window.setCentralWidget(mdi)
    from .active_window import follow_active_window, follow_panel_focus

    follow_active_window(mdi)  # the window in front: an accent title bar and frame; the others' titles muted
    follow_panel_focus(window)  # the panel with the keyboard focus: an accent line along its top

    objects_box = Qt.QWidget()
    objects_layout = Qt.QVBoxLayout(objects_box)
    objects_layout.setContentsMargins(0, 0, 0, 0)
    object_filter = Qt.QLineEdit()
    object_filter.setPlaceholderText("Filter by name or type")
    object_filter.setClearButtonEnabled(True)
    objects_layout.addWidget(object_filter)
    class ObjectsTree(Qt.QTreeWidget):
        def resizeEvent(self, event: Any) -> None:  # noqa: N802
            super().resizeEvent(event)
            fit_object_columns()
            place_empty()

        def viewportEvent(self, event: Any) -> bool:  # noqa: N802
            handled = super().viewportEvent(event)
            if event.type() == QEvent.Type.Resize:  # narrower without the panel resizing: a scroll bar came
                fit_object_columns()
            return handled

    objects = ObjectsTree()
    objects.setHeaderLabels(["Name", "Type", "Status"])
    objects.setRootIsDecorated(False)
    objects.setColumnHidden(2, True)  # Status shows once a result is out of date, not computed or frozen
    objects.setHorizontalScrollBarPolicy(QtNS.ScrollBarPolicy.ScrollBarAlwaysOff)  # long names are shortened instead
    size_icons(objects)  # the kinds' icons at the text size (View → Text size)
    from PySide6.QtGui import QPalette

    objects_empty = Qt.QLabel(objects.viewport())  # the empty state inside the list, at its top, as History's
    objects_empty.setWordWrap(True)
    objects_empty.setMargin(6)
    objects_empty.setForegroundRole(QPalette.ColorRole.PlaceholderText)
    objects_empty.setAttribute(QtNS.WidgetAttribute.WA_TransparentForMouseEvents)

    def place_empty() -> None:
        width = objects.viewport().width()
        objects_empty.setGeometry(0, 0, width, objects_empty.heightForWidth(width))

    def fit_object_columns() -> None:
        """Name and Type as wide as their contents ("Posterior (Metropolis-Hastings)" in full), Status the rest; in a
        narrow panel a long name gives way (shortened with "…", in full in its tooltip) so that Type stays in view."""
        if not objects.topLevelItemCount():
            return
        objects.resizeColumnToContents(0)
        objects.resizeColumnToContents(1)
        header = objects.header()
        room = objects.viewport().width() - header.sectionSize(1)
        if header.sectionSize(0) > room:
            header.resizeSection(0, max(least_name_width(), room))

    def least_name_width() -> int:
        """How narrow the Name column may get for Type to stay in view: 80 pixels, or its contents if narrower."""
        header = objects.header()
        return min(max(objects.sizeHintForColumn(0), header.sectionSizeHint(0)), max(80, header.sectionSizeHint(0)))

    def objects_panel_width() -> int:
        """The Objects panel's width with Name (at its least) and Type in full: the columns, the scroll bar, frame
        and margins, as wide as this font and style make them (0 while the panel shows nothing)."""
        if not objects.topLevelItemCount() or not objects_dock.isVisible():
            return 0
        chrome = objects_dock.width() - objects.viewport().width()  # margins, frame and a scroll bar if shown
        if not objects.verticalScrollBar().isVisible():  # room for one, should the list grow
            chrome += objects.style().pixelMetric(Qt.QStyle.PixelMetric.PM_ScrollBarExtent)
        return least_name_width() + objects.sizeHintForColumn(1) + chrome

    def apply_filter(text: str = "") -> None:
        words = (text or object_filter.text()).lower().split()
        shown = 0
        for i in range(objects.topLevelItemCount()):
            item = objects.topLevelItem(i)
            haystack = f"{item.text(0)} {item.text(1)} {item.text(2)}".lower()
            item.setHidden(not all(w in haystack for w in words))
            shown += not item.isHidden()
        if not objects.topLevelItemCount():
            objects_empty.setText(EMPTY_STATES["objects"])
        elif not shown:
            objects_empty.setText(f"No objects match “{object_filter.text()}”.")
        objects_empty.setVisible(not shown)
        place_empty()

    object_filter.textChanged.connect(apply_filter)
    recompute_button = Qt.QPushButton("Recalculate")
    recompute_button.setToolTip("re-run the steps that made results whose inputs have changed since, and compute deferred ones")

    class _CalcMode:  # the Calculate menu's mode, kept in step with the session (set once the menu exists)
        @staticmethod
        def update(_session: Any) -> None:
            return None

    calc_mode = _CalcMode()
    recompute_button.setVisible(False)
    objects_layout.addWidget(objects, 1)
    objects_layout.addWidget(recompute_button)
    objects_dock = Dock("Objects", window)
    objects_dock.setObjectName("objects")
    objects_dock.setWidget(objects_box)
    window.addDockWidget(QtNS.DockWidgetArea.LeftDockWidgetArea, objects_dock)

    from .history_qt import HistoryPanel

    history = HistoryPanel(lambda code: send_immediate(code), lambda code: send_editor(code),
                           lambda records: rerun(records), lambda record: reopen(record))
    history_dock = Dock("History", window)
    history_dock.setObjectName("history")
    history_dock.setWidget(history)
    window.addDockWidget(QtNS.DockWidgetArea.LeftDockWidgetArea, history_dock)

    script = Qt.QPlainTextEdit()
    script.setReadOnly(True)
    from .text_size import fixed_font

    script.setFont(fixed_font())  # Python as in the code editor, at the app's text size
    script.setPlaceholderText(EMPTY_STATES["script"])

    def script_menu(pos: Any) -> Any:
        """The Script panel's right-click menu: the standard copy actions, then the whole script copied or opened in a
        Code window, to change and run."""
        menu = script.createStandardContextMenu(pos)
        menu.addSeparator()
        menu.addAction("Copy the whole script", lambda: Qt.QApplication.clipboard().setText(script.toPlainText()))
        menu.addAction("Open in a Code window", lambda: open_editor(text=current().script()))
        return menu

    script.setContextMenuPolicy(QtNS.ContextMenuPolicy.CustomContextMenu)
    script.customContextMenuRequested.connect(lambda pos: script_menu(pos).popup(script.viewport().mapToGlobal(pos)))
    window.script_menu = script_menu
    script_dock = Dock("Script", window)
    script_dock.setObjectName("script")
    script_dock.setWidget(script)
    window.addDockWidget(QtNS.DockWidgetArea.BottomDockWidgetArea, script_dock)

    from . import console_qt
    from .editor_qt import EditorWindow, error_line

    immediate: console_qt.Immediate | None = None
    running: dict[str, Any] = {}  # the Code window whose code the Immediate window is running, and its first line
    immediate_dock = Dock("Immediate", window)
    immediate_dock.setObjectName("immediate")
    if console_qt.available():
        immediate = console_qt.Immediate(session, lambda record: after_code(record))
        immediate_dock.setWidget(immediate.widget)
    else:
        immediate_dock.setWidget(Qt.QLabel("  The Immediate window needs qtconsole:  pip install pymetrix[qt]"))
    window.addDockWidget(QtNS.DockWidgetArea.BottomDockWidgetArea, immediate_dock)
    window.tabifyDockWidget(script_dock, immediate_dock)
    problems_view = Qt.QTreeWidget()  # what needs attention: out-of-date results, cautions, failures, warnings
    problems_view.setHeaderLabels(["Object or step", "Problem"])
    problems_view.setRootIsDecorated(False)
    problems_dock = Dock("Problems", window)
    problems_dock.setObjectName("problems")
    problems_dock.setWidget(problems_view)
    window.addDockWidget(QtNS.DockWidgetArea.BottomDockWidgetArea, problems_dock)
    window.tabifyDockWidget(immediate_dock, problems_dock)
    from .report_qt import report_panel

    report_view = report_panel(lambda: current(), lambda: export_report_outline(), lambda: update_title(), show_now=False,
                               open_result=lambda n: show_object(n))
    report_dock = Dock("Report", window)
    report_dock.setObjectName("report")
    report_dock.setWidget(report_view)
    window.addDockWidget(QtNS.DockWidgetArea.RightDockWidgetArea, report_dock)
    report_dock.setVisible(False)  # the Report workspace shows it; View → Workspace → Report
    immediate_dock.raise_()
    status_labels = {key: Qt.QLabel() for key in ("dataset", "sample", "selected", "last")}
    for key in ("dataset", "sample", "selected", "last"):  # a quiet summary, always there
        status_labels[key].setStyleSheet("padding: 0 8px;")
        window.statusBar().addPermanentWidget(status_labels[key])

    def message_room(text: str) -> None:
        """While the status bar says something (a step's outcome, a demo's guide), the last step's line, which says
        much the same, gives it the room: a message was cut off by the four fields at the right."""
        status_labels["last"].setVisible(not text)

    window.statusBar().messageChanged.connect(message_room)

    progress_box = Qt.QWidget()
    progress_row = Qt.QHBoxLayout(progress_box)
    progress_row.setContentsMargins(0, 0, 0, 0)
    progress_label = Qt.QLabel()
    progress_bar = Qt.QProgressBar()
    progress_bar.setAccessibleName("Step progress")
    progress_bar.setMaximumWidth(220)
    progress_bar.setRange(0, 1000)
    cancel_button = Qt.QPushButton("Cancel")
    cancel_button.setToolTip("stop the running step at its next check; nothing is recorded")
    for widget in (progress_label, progress_bar, cancel_button):
        progress_row.addWidget(widget)
    window.statusBar().addWidget(progress_box, 1)
    progress_box.setVisible(False)
    running_task: dict[str, Any] = {}

    def progress_show(task_: Any, label: str) -> None:
        running_task["task"] = task_
        progress_label.setText(f"Running {label or 'the step'}…")
        cancel_button.setEnabled(not task_.cancelled)  # Cancel pressed (in its dialog) before the bar showed: say so
        cancel_button.setText("Cancelling…" if task_.cancelled else "Cancel")
        progress_box.setVisible(True)
        immediate_dock.widget().setEnabled(False)  # typed code waits: the session is in use
        progress_update(task_)
        if task_.cancelled:
            progress_label.setText("Cancelling: the step stops at its next check")

    def progress_update(task_: Any) -> None:
        if task_.fraction is None:
            progress_bar.setRange(0, 0)  # moving, length unknown
        else:
            progress_bar.setRange(0, 1000)
            progress_bar.setValue(int(1000 * task_.fraction))
        if task_.message and not task_.cancelled:
            progress_label.setText(task_.message)

    def progress_hide() -> None:
        progress_box.setVisible(False)
        immediate_dock.widget().setEnabled(True)
        running_task.clear()

    def cancel_step() -> None:
        # the step running now, whether or not its bar has shown yet (it shows on the first tick after 0.4 s)
        task_ = running_task.get("task") or _BUSY.get("task")
        if task_ is not None:
            task_.cancel()
            cancel_button.setEnabled(False)
            cancel_button.setText("Cancelling…")
            progress_label.setText("Cancelling: the step stops at its next check")

    cancel_button.clicked.connect(cancel_step)
    _PROGRESS.clear()  # this window's bar from now on, and only this window clears it when it closes
    _PROGRESS.update(show=progress_show, update=progress_update, hide=progress_hide, cancel=cancel_step, window=window)
    window.progress_box, window.cancel_step = progress_box, cancel_step

    default_state = window.saveState()

    def fit_default_layout() -> None:
        """Panels in a default arrangement (a first run, a workspace never arranged, Reset layout) sized to the window,
        by ``default_dock_size``: the result windows keep most of the height, the Objects panel shows names and types.
        Done again as the window resizes, until the user drags a panel's edge or moves a panel; a saved arrangement
        is never touched."""
        layout_fit["pending"] = True
        shown = [d for d in window.findChildren(Qt.QDockWidget) if d.isVisible() and not d.isFloating()]
        bottom = [d for d in shown if window.dockWidgetArea(d) == QtNS.DockWidgetArea.BottomDockWidgetArea]
        left = [d for d in shown if window.dockWidgetArea(d) == QtNS.DockWidgetArea.LeftDockWidgetArea]
        if bottom:
            # the window's own bars (menus, toolbars, status bar) taken into account: with a large font or style they
            # are taller, and the bottom panels give way so that the result windows keep three fifths of the height
            bars = [window.menuBar(), window.statusBar(), *(t for t in window.findChildren(Qt.QToolBar)
                                                            if t.isVisible() and not t.isFloating()
                                                            and window.toolBarArea(t) in (QtNS.ToolBarArea.TopToolBarArea,
                                                                                          QtNS.ToolBarArea.BottomToolBarArea))]
            chrome = sum(max(bar.height(), bar.sizeHint().height()) for bar in bars if bar.isVisible())
            room = round(0.4 * window.height()) - chrome - 2 * window.style().pixelMetric(Qt.QStyle.PixelMetric.PM_DockWidgetSeparatorExtent)
            size = max(90, min(default_dock_size("bottom", window.height(), window.width()), room))
            window.resizeDocks(bottom, [size] * len(bottom), QtNS.Orientation.Vertical)
        if left:
            size = default_dock_size("left", window.height(), window.width())
            size = max(size, min(objects_panel_width(), round(0.4 * window.width())))  # Type in full, with this font
            window.resizeDocks(left[:1], [size], QtNS.Orientation.Horizontal)

    def user_layout(*_: Any) -> None:
        layout_fit["pending"] = False

    for dock_ in window.findChildren(Qt.QDockWidget):  # a panel moved, floated or docked elsewhere: the user's layout
        dock_.topLevelChanged.connect(user_layout)
    window.fit_default_layout, window.layout_fit = fit_default_layout, layout_fit
    open_windows: dict[str, Any] = {}

    def current() -> Session:
        return state["session"]

    def update_title() -> None:
        s = current()
        name = s.path.name if s.path else (f"demo: {s.demo}" if getattr(s, "demo", None) else "untitled")
        window.setWindowTitle(f"{NAME} — {name}{' *' if s.dirty else ''}")

    def selected_name() -> str | None:
        item = objects.currentItem()
        return item.text(0) if item is not None and item.text(0) in current().objects else None

    def run_for_plugin(command: Any) -> Any:
        """A plugin's step, run as the dialogs run theirs: refused while one runs, in the background, then refreshed."""
        if busy():
            raise RuntimeError("a step is running: wait for it, or cancel it")
        result = run_blocking(lambda: current().run(command), type(command).__name__)
        refresh()
        return result

    from .extend import AppInterface

    app_interface = AppInterface(lambda: current(), selected_name, lambda cls, initial=None: run_dialog(cls, initial=initial),
                                 run_for_plugin, lambda text: window.statusBar().showMessage(text, 10000))
    window.app_interface = app_interface

    def install_panels() -> dict[str, Any]:
        """Docks for the plugins' panels (``panels``), each given the app's interface; a panel that fails to build is
        a problem, not a crash."""
        from ..plugins import registry
        from .extend import AppPanel

        out: dict[str, Any] = {}
        areas = {"left": QtNS.DockWidgetArea.LeftDockWidgetArea, "right": QtNS.DockWidgetArea.RightDockWidgetArea,
                 "bottom": QtNS.DockWidgetArea.BottomDockWidgetArea}
        for entry in registry("panels").entries(available=True):
            spec = entry.obj
            if not isinstance(spec, AppPanel) or spec.title in out:
                continue
            try:
                widget = spec.create(app_interface)
            except Exception as exc:  # noqa: BLE001 - say so in the status bar and carry on
                window.statusBar().showMessage(f"The panel {spec.title} ({entry.plugin.name}) could not be built: {exc}", 15000)
                continue
            dock = Dock(spec.title, window)
            dock.setObjectName(f"plugin:{entry.qualified}")
            dock.setWidget(widget)
            dock.setToolTip(f"from the plugin {entry.plugin.name}")
            window.addDockWidget(areas[spec.area], dock)
            dock.setVisible(False)  # its View toggle and its workspace show it
            view.addAction(dock.toggleViewAction())
            out[spec.title] = dock
        window.plugin_docks = out
        return out

    def update_problems() -> None:
        """Problems: out-of-date and not-computed results, steps that used them or failed, and results' own warnings."""
        s = current()
        problems_view.clear()
        found = [(n, f"out of date: {why}") for n, why in s.stale().items()]
        found += [(n, "not computed yet (deferred: a slow step when the project was opened, or a step added deferred)"
                   + (f"; it may not compute: {r.note}" if r.note else "")) for n, r in s.pending().items()]
        refers = [("stale", n) for n in s.stale()] + [("stale", n) for n in s.pending()]  # what each line is about
        from ..plugins import load_problems

        found += [(f"plugin {name}", f"could not load: {why} (Help → Plugins)") for name, why in load_problems().items()]
        found += [(f"plugin {name}", f"needed by this project: {why}") for name, why in getattr(s, "missing_plugins", {}).items()]
        refers += [("plugin", name) for name in load_problems()] + [("plugin", name) for name in getattr(s, "missing_plugins", {})]
        resolved = s.resolved_failures()  # failed, then run again successfully: kept in History (greyed), not a problem
        first = max(0, len(s.history) - 200)
        for i, record in enumerate(s.history[first:], first):
            step = record.code.splitlines()[0][:60] if record.code else ""
            if record.skipped and record.error:
                found.append((step, record.error))  # kept, not run, and why
                refers.append(("step", i))
            elif not record.ok and record.error and i not in resolved:
                found.append((step, f"failed: {record.error}"))
                refers.append(("step", i))
            for caution in getattr(record, "cautions", ()) or ():
                found.append((step, f"used {caution}"))
                refers.append(("step", i))
        for n, obj in s.objects.items():
            for warning in getattr(getattr(obj, "provenance", None), "warnings", ()) or ():
                found.append((n, f"warning: {warning}"))
                refers.append(("object", n))
        for (where, what), about in zip(found, refers):
            item = Qt.QTreeWidgetItem([where, what])
            item.setToolTip(1, what)
            item.setData(0, QtNS.ItemDataRole.UserRole, about)
            problems_view.addTopLevelItem(item)
        if not found:  # an empty state across the row, as in History
            empty = Qt.QTreeWidgetItem([EMPTY_STATES["problems"]])
            empty.setToolTip(0, EMPTY_STATES["problems"])
            empty.setFlags(QtNS.ItemFlag.NoItemFlags)
            problems_view.addTopLevelItem(empty)
            empty.setFirstColumnSpanned(True)
        problems_view.resizeColumnToContents(0)
        problems_dock.setWindowTitle(f"Problems ({len(found)})" if found else "Problems")
        window.problem_list = found

    def problems_menu(item: Any) -> Any:
        """The right-click menu on a line of Problems, for what it is about: a step (show it in History, open its
        dialog filled in, copy its code), a result out of date (open it, recalculate it), a result's warning (open it)
        or a plugin (Help → Plugins); and copy the problem's words."""
        menu = Qt.QMenu(problems_view)
        about = item.data(0, QtNS.ItemDataRole.UserRole) if item is not None else None
        if about is None:
            return menu
        kind, key = about
        s = current()
        if kind == "step" and 0 <= key < len(s.history):
            record = s.history[key]
            menu.addAction("Show the step in History", lambda: (history_dock.show(), history_dock.raise_(), history.select_record(record)))
            menu.addAction("Open its dialog, filled in…", lambda: reopen(record)).setEnabled(not record.code_step)
            menu.addAction("Copy its code", lambda: Qt.QApplication.clipboard().setText(record.code or ""))
        elif kind == "stale":
            menu.addAction(f"Open “{key}”", lambda: show_object(key)).setEnabled(key in s.objects)
            menu.addAction(f"Recalculate “{key}”", lambda: recompute([key]))
            menu.addAction("Recalculate everything out of date", lambda: recompute())
        elif kind == "object":
            menu.addAction(f"Open “{key}”", lambda: show_object(key)).setEnabled(key in s.objects)
        elif kind == "plugin":
            menu.addAction("Plugins…", lambda: show_plugins())
        menu.addSeparator()
        menu.addAction("Copy the problem", lambda: Qt.QApplication.clipboard().setText(f"{item.text(0)}: {item.text(1)}"))
        return menu

    def problems_default(item: Any) -> None:
        """A double-click: the menu's first action (open the dialog, show the result...)."""
        actions = [a for a in problems_menu(item).actions() if not a.isSeparator() and a.isEnabled()]
        if actions:
            actions[0].trigger()

    problems_view.setContextMenuPolicy(QtNS.ContextMenuPolicy.CustomContextMenu)
    problems_view.customContextMenuRequested.connect(
        lambda pos: problems_menu(problems_view.itemAt(pos)).popup(problems_view.viewport().mapToGlobal(pos))
        if problems_view.itemAt(pos) is not None else None)
    problems_view.itemDoubleClicked.connect(lambda item, _col: problems_default(item))
    window.problems_menu = problems_menu

    def update_status() -> None:
        """The status bar: the dataset in focus with its frequency and sample, the selected object, the last step."""
        from ..data import Dataset

        s = current()
        item = objects.currentItem()
        chosen = item.text(0) if item is not None and item.text(0) in s.objects else None
        datasets = [n for n, o in s.objects.items() if isinstance(o, Dataset)]
        focus = chosen if chosen in datasets else ("data" if "data" in datasets else (datasets[0] if datasets else None))
        if focus is None:
            status_labels["dataset"].setText("No dataset")
            status_labels["sample"].setText("")
            status_labels["sample"].setVisible(False)  # no empty bordered cell (the Windows style draws one)
        else:
            status_labels["sample"].setVisible(True)
            data = s.objects[focus]
            index = data.frame.index
            freq = (getattr(index, "freqstr", "") or "").split("-")[0].rstrip("ES")
            words = {"Q": "quarterly", "M": "monthly", "Y": "annual", "A": "annual", "W": "weekly", "D": "daily", "B": "business daily"}
            status_labels["dataset"].setText(f"{focus}: {len(data.names)} series" + (f", {words.get(freq, freq)}" if freq else ""))
            status_labels["sample"].setText(f"sample {data.sample}" if data.meta.sample is not None
                                            else "all periods" if freq else "all rows")
        status_labels["selected"].setText(f"{chosen} ({kind_of(s.objects[chosen])})" if chosen else "No object selected")
        last = s.history[-1] if s.history else None
        if last is None:
            status_labels["last"].setText("No steps yet")
        else:
            mark = "✓" if last.ok else "✗ failed"
            first = last.code.splitlines()[0] if last.code else ""
            status_labels["last"].setText(f"Last step {mark}: {first if len(first) <= 40 else first[:39] + '…'}")
            status_labels["last"].setToolTip(last.code + (f"\n{last.error}" if not last.ok and last.error else ""))

    objects.currentItemChanged.connect(lambda *_: update_status())

    def follow_active(sub: Any) -> None:
        """The window brought to the front is the object in focus: selected in Objects, named in the status bar."""
        name = next((n for n, w in open_windows.items() if w is sub), None)
        if name is None:
            return
        found = objects.findItems(name, QtNS.MatchFlag.MatchExactly, 0)
        if found and objects.currentItem() is not found[0]:
            objects.setCurrentItem(found[0])

    mdi.subWindowActivated.connect(follow_active)

    def producer_code(name: str) -> str:
        record = current().producer(name)
        return record.code if record is not None else ""

    def fit_in_workspace(sub: Any) -> None:
        """Place a new window in cascade inside the workspace. Qt's own placement can put it partly off the right or
        bottom edge on a laptop screen, or exactly over the last one. Shrunk to fit if the workspace is small."""
        area = mdi.viewport().rect()
        if area.width() < 50 or area.height() < 50:  # not laid out yet
            return
        cascade = 28  # each window down and right of the last, so the title bars stay visible
        room = 3 * cascade if area.width() > 6 * cascade and area.height() > 6 * cascade else 0
        sub.resize(min(sub.width(), area.width() - room), min(sub.height(), area.height() - room))
        step = cascade * max(len(mdi.subWindowList()) - 1, 0)
        room_x, room_y = max(area.width() - sub.width(), 0), max(area.height() - sub.height(), 0)
        sub.move(step % (room_x + 1), step % (room_y + 1))

    def show_object(name: str, activate: bool = True) -> None:
        from ..data import Dataset

        obj = current()[name]
        title = window_title(name, obj)
        shown_obj, scope = obj, ""
        actions = series_actions_for(name) if isinstance(obj, Dataset) else None
        names = current().working_set(name)
        if names is not None:  # a dataset with a working set: its window shows those series
            shown_obj = _scoped_dataset(obj, names)
            scope = f"{len(names):,} of {len(obj.names):,} series (working set: {current().working_sets[name]})"
        sub = open_windows.get(name)
        if sub is not None and getattr(sub, "shown", None) is obj and getattr(sub, "scope", "") == scope:
            pass  # already showing this object, as it is now
        elif sub is not None:
            old = sub.widget()
            sub.setWidget(object_view(shown_obj, name=name, code=producer_code(name), edit_figure=lambda t, f, k="options", n=name: edit_figure(n, t, f, k),
                                      place_note=lambda t, f, panel, x, y, n=name: place_note(n, t, f, panel, x, y),
                                      heatmap=lambda t, n=name: table_heatmap(n, t), scope=scope, series_actions=actions,
                                      add_to_report=lambda t, n=name: add_to_report(n, t), next_steps=next_steps_for(name),
                                      matrix_menu=matrix_menu_for(name), figure_edits=lambda *a, n=name: figure_edit(n, *a)))
            if getattr(old, "zoom_level", None) not in (None, 3):  # the new view keeps the window's zoom
                sub.widget().zoom_to(old.zoom_level)
            if hasattr(old, "current_title") and hasattr(sub.widget(), "select"):  # and the view it was showing
                sub.widget().select(old.current_title())
            old.deleteLater()
            sub.setWindowTitle(title)
            sub.setWindowIcon(object_icon(obj))
        else:
            sub = mdi.addSubWindow(object_view(shown_obj, name=name, code=producer_code(name),
                                               edit_figure=lambda t, f, k="options", n=name: edit_figure(n, t, f, k),
                                      place_note=lambda t, f, panel, x, y, n=name: place_note(n, t, f, panel, x, y),
                                      heatmap=lambda t, n=name: table_heatmap(n, t), scope=scope, series_actions=actions,
                                      add_to_report=lambda t, n=name: add_to_report(n, t), next_steps=next_steps_for(name),
                                      matrix_menu=matrix_menu_for(name), figure_edits=lambda *a, n=name: figure_edit(n, *a)))
            sub.setAttribute(QtNS.WidgetAttribute.WA_DeleteOnClose)
            sub.setWindowTitle(title)
            sub.setWindowIcon(object_icon(obj))
            area = mdi.viewport().rect()  # most of the workspace, cascaded (fit_in_workspace), not a fixed 680 by 460
            sub.resize(max(680, min(round(0.85 * area.width()), 1100)), max(460, min(round(0.9 * area.height()), 760)))
            open_windows[name] = sub
            sub.destroyed.connect(lambda _=None, n=name: open_windows.pop(n, None))
            sub.show()
            fit_in_workspace(sub)
        sub.shown = obj
        sub.scope = scope
        if activate:
            sub.showNormal()
            mdi.setActiveSubWindow(sub)

    def refresh() -> None:
        from PySide6.QtGui import QBrush

        s = current()
        report_view.refresh()  # the outline of this session (another project's, once opened)
        if "app_interface" in window.__dict__:
            window.app_interface._changed()  # plugins' panels follow the history
        if hasattr(window, "update_undo"):
            window.update_undo()
        if hasattr(window, "update_badges"):  # each stage's results needing attention, on the stage bar
            from .workspaces import stage_of, stage_of_command

            attention: dict[str, list[str]] = {}
            pending = s.pending()
            for name in set(s.stale()) | set(pending):
                stage = stage_of(s.objects[name]) if name in s.objects else                     stage_of_command(pending[name].command) if name in pending else "estimate"
                attention.setdefault(stage, []).append(name)
            window.update_badges(attention)
        objects.clear()
        stale, frozen_stale, pending = s.stale(), s.stale(frozen=True), s.pending()
        for name in sorted(set(s.objects) | set(pending)):
            if name in pending:
                status, tip, colour = "not computed", "Not computed yet (deferred: a slow step when the project was opened, or a step added deferred). Compute it from the menu, or Recalculate.", "muted"
            elif name in stale:
                status, tip, colour = "out of date", f"Out of date: {stale[name]}. Recalculate to update it.", "warning"
            elif name in s.frozen:
                status = "frozen; inputs changed" if name in frozen_stale else "frozen"
                tip, colour = "Frozen: never recomputed until you unfreeze it." + (f" ({frozen_stale[name]})" if name in frozen_stale else ""), "link"
            else:
                status, tip, colour = "", "", None
            item = Qt.QTreeWidgetItem([name, kind_of(s.objects[name]) if name in s.objects else "", status])
            item.setIcon(0, object_icon(s.objects[name]) if name in s.objects else kind_icon("generic"))  # decoration: the text is the same
            if not colour:
                item.setToolTip(0, name)  # in full, where a narrow panel shortens it
            if colour:
                item.setForeground(2, QBrush(theme.colour(colour)))
                for col in range(3):
                    item.setToolTip(col, tip)
            objects.addTopLevelItem(item)
        objects.setColumnHidden(2, not any(objects.topLevelItem(i).text(2) for i in range(objects.topLevelItemCount())))
        apply_filter()
        fit_object_columns()
        todo = len(stale) + len(pending)
        recompute_button.setVisible(bool(todo))
        recompute_button.setText(f"Recalculate {todo} result{'s' if todo != 1 else ''} (F9)")
        calc_mode.update(s)
        script.setPlainText(s.script() if s.history else "")
        history.show_session(s)
        update_problems()
        update_status()
        for name in list(open_windows):  # windows of objects that changed or disappeared
            if name in s.objects:
                show_object(name, activate=False)
            else:
                open_windows[name].close()
        for name, sheet in list(sheets.items()):  # spreadsheets follow their datasets
            if not sheet.sync():
                sheet.model.pending.clear()
                sheet.parentWidget().close()
        update_title()

    def activate(name: str) -> None:
        if name in current().pending():
            if Qt.QMessageBox.question(window, "Not computed", f"{name} is not computed yet. Compute it now?") != Qt.QMessageBox.StandardButton.Yes:
                return
            compute([name])
        if name in current().objects:
            show_object(name)

    objects.itemActivated.connect(lambda item, _col: activate(item.text(0)))

    def run_dialog(cls: type[Command], initial: Command | None = None) -> None:
        if cls.__name__ == "DefineModel":  # a model as text: its own window, checked as it is typed
            open_model_editor(initial)
            return
        if cls.__name__ == "EnterMatrix":  # numbers in a grid, typed or pasted
            open_matrix_grid(initial)
            return
        if busy():
            Qt.QMessageBox.information(window, "A step is running", "Wait for the running step to finish, or Cancel it "
                                       "(in the status bar), before starting another.")
            return
        dialog = command_dialog(cls, current(), window, initial=initial)
        accepted = dialog.exec()
        if not accepted and getattr(dialog, "open_instead", None):  # its picker's button: the dialog that makes one
            run_dialog(dialog.open_instead[0])
            return
        if accepted:
            refresh()
            last = current().history[-1] if current().history else None
            if last is not None and last.ok and last.cautions:  # it used results that were not up to date
                Qt.QMessageBox.warning(window, "Used results that are not up to date",
                                       "This step used:\n\n" + "\n".join(f"• {c}" for c in last.cautions)
                                       + "\n\nRecalculate (F9) and run it again for up-to-date results.")
            name = getattr(dialog, "command", None) and getattr(dialog.command, "name", None)
            if name in current().objects:
                show_object(name)
            from pathlib import Path as _Path

            if isinstance(dialog.result_value, _Path):  # a file written: say where
                window.statusBar().showMessage(f"Saved {dialog.result_value.resolve()}", 15000)

    def open_series_browser() -> Any:
        """File → Browse series…: find series on a data service; Load records a Load from connector step, and Add to
        Load from connector opens that dialog with the codes filled in."""
        from .series_browser_qt import series_browser

        def loaded(name: str) -> None:
            refresh()
            if name in current().objects:
                show_object(name)

        def edit(cmd: Any) -> None:
            browser.dialog.accept()
            run_dialog(type(cmd), initial=cmd)

        browser = series_browser(current(), window, on_load=loaded, on_edit=edit)
        window.series_browser = browser  # for scripting and tests
        browser.dialog.show()
        return browser

    window.open_series_browser = open_series_browser

    def open_refresh_data() -> Any:
        """File → Refresh data…: read datasets again from their files or data services. A project keeps the data as
        read, so this is the one way new or revised data comes in (Session.refresh)."""
        from PySide6.QtCore import Qt as QtNS

        s = current()
        names = s.refreshable()
        if not names:
            Qt.QMessageBox.information(window, "Refresh data", "No dataset here was read from a file or a data service "
                                       "(File → Load data… or File → Load from connector…), so there is nothing to refresh.")
            return None
        dialog = Qt.QDialog(window)
        dialog.setWindowTitle("Refresh data")
        layout = Qt.QVBoxLayout(dialog)
        layout.addWidget(Qt.QLabel("Read these datasets again from their sources. The steps that changed them are run "
                                   "again, and results computed from them are recomputed (Automatic) or marked out of date "
                                   "(Manual). A load pinned to a vintage reads that vintage again."))
        listing = Qt.QListWidget(dialog)
        for name in names:
            source = s._load_step(name)
            item = Qt.QListWidgetItem(f"{name}    {source.code.strip().splitlines()[0] if source else ''}")
            item.setData(QtNS.ItemDataRole.UserRole, name)
            item.setFlags(item.flags() | QtNS.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtNS.CheckState.Checked)
            listing.addItem(item)
        layout.addWidget(listing)
        buttons = Qt.QDialogButtonBox(Qt.QDialogButtonBox.StandardButton.Ok | Qt.QDialogButtonBox.StandardButton.Cancel, dialog)
        buttons.button(Qt.QDialogButtonBox.StandardButton.Ok).setText("Refresh")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        def chosen() -> list[str]:
            return [listing.item(i).data(QtNS.ItemDataRole.UserRole) for i in range(listing.count())
                    if listing.item(i).checkState() == QtNS.CheckState.Checked]

        def run() -> None:
            picked = chosen()
            if picked:
                try:
                    current().refresh(*picked)
                except Exception as exc:
                    Qt.QMessageBox.critical(window, *titled("Could not refresh", str(exc)))
            refresh()

        dialog.accepted.connect(run)
        dialog.listing, dialog.chosen = listing, chosen  # for scripting and tests
        window.refresh_data_dialog = dialog
        dialog.show()
        return dialog

    window.open_refresh_data = open_refresh_data

    # recomputing, the history, the Immediate and Code windows

    def recompute(names: Any = None) -> None:
        """Recalculate: deferred results first (all, or the named ones), then out-of-date results, in dependency order."""
        s = current()
        try:
            if s.pending() and (names is None or set(names) & set(s.pending())):
                s.compute(names)
            s.recompute(names)
        except Exception as exc:
            Qt.QMessageBox.critical(window, *titled("Could not recalculate", str(exc)))
        refresh()

    def compute(names: Any = None) -> None:
        try:
            current().compute(names)
        except Exception as exc:
            Qt.QMessageBox.critical(window, *titled("Could not compute", str(exc)))
        refresh()

    def freeze(name: str, on: bool) -> None:
        current().freeze(name) if on else current().unfreeze(name)
        refresh()

    def clean_up_history() -> None:
        dead = current().dead_ends()
        if not dead:
            Qt.QMessageBox.information(window, "Clean up history", "Every step in the history leads to a current result: nothing to remove.")
            return
        listing = "\n".join(f"• {r.code.splitlines()[0][:90]}" for r in dead[:15]) + ("\n…" if len(dead) > 15 else "")
        answer = Qt.QMessageBox.question(window, "Clean up history",
                                         f"Remove {len(dead)} step{'s' if len(dead) != 1 else ''} that no current result depends on "
                                         f"(results later replaced or deleted)? They leave the history and the script.\n\n{listing}")
        if answer == Qt.QMessageBox.StandardButton.Yes:
            current().prune()
            refresh()

    recompute_button.clicked.connect(lambda: recompute())

    # the spreadsheet editor

    sheets: dict[str, Any] = {}

    def run_step(command: Command) -> bool:
        """Record a step from a window (spreadsheet edits); errors are shown, not raised."""
        try:
            current().run(command)
        except Exception as exc:
            Qt.QMessageBox.critical(window, *titled("Could not make the change", str(exc)))
            refresh()
            return False
        refresh()
        return True

    def open_model_editor(initial: Command | None = None, translation: Any = None) -> Any:
        """Model > Model editor: a new model, or the text of an earlier DefineModel step to change (or a model written
        in another syntax, with its translation notes)."""
        from .model_editor_qt import ModelEditorWindow

        def apply(command: Command) -> bool:
            if not run_step(command):
                return False
            if command.name in current().objects:
                show_object(command.name)
            return True

        def open_other(text: str, syntax: str, name: str, written: Any) -> Any:
            from types import SimpleNamespace

            return open_model_editor(SimpleNamespace(text=text, syntax=syntax, name=name), written)

        editor = ModelEditorWindow(apply, text=getattr(initial, "text", ""), syntax=getattr(initial, "syntax", "pymetrix"),
                                   name=getattr(initial, "name", "model"), open_other=open_other, translation=translation)
        sub = Qt.QMdiSubWindow()
        sub.setWidget(editor)
        sub.setAttribute(QtNS.WidgetAttribute.WA_DeleteOnClose)
        sub.setWindowTitle(f"{editor.name.text()} — Model editor")  # name — kind, as every window
        sub.setWindowIcon(kind_icon("model"))
        mdi.addSubWindow(sub)
        sub.resize(980, 560)
        sub.show()
        fit_in_workspace(sub)
        editor.editor.setFocus()
        return editor

    window.open_model_editor = open_model_editor

    def open_sheet(name: str | None = None) -> Any:
        from ..data import Dataset
        from .sheet_qt import SheetWindow

        if name is None:
            names = current().of_type("Dataset")
            if not names:
                Qt.QMessageBox.information(window, "Spreadsheet", "There is no dataset yet: load one with File → Load data.")
                return None
            if len(names) == 1:
                name = names[0]
            else:
                name, ok = Qt.QInputDialog.getItem(window, "Spreadsheet", "Dataset:", names, 0, False)
                if not ok:
                    return None
        if not isinstance(current().objects.get(name), Dataset):
            return None
        if name in sheets:
            mdi.setActiveSubWindow(sheets[name].parentWidget())
            return sheets[name]
        def scope_of(n: str = name) -> Any:
            names = current().working_set(n)
            return None if names is None else (names, current().working_sets[n])

        sheet = SheetWindow(name, lambda n=name: current()[n], run_step, working_set=scope_of,
                            series_actions=series_actions_for(name))

        class SheetSub(Qt.QMdiSubWindow):
            def closeEvent(self, event: Any) -> None:
                if sheet.confirm_close():
                    sheets.pop(name, None)
                    super().closeEvent(event)
                else:
                    event.ignore()

        sub = SheetSub()
        sub.setWidget(sheet)
        sub.setWindowIcon(kind_icon("dataset"))
        sub.setAttribute(QtNS.WidgetAttribute.WA_DeleteOnClose)
        mdi.addSubWindow(sub)
        sub.resize(860, 520)
        sheets[name] = sheet
        sheet.update_status()
        sub.show()
        fit_in_workspace(sub)
        return sheet

    def series_actions_for(dataset: str) -> dict[str, Any]:
        """What a dataset's series browser does with the series picked in it."""
        from ..commands import GenerateSeries, PlotSeries, SelectSeries

        def plot(names: list[str]) -> None:
            run_dialog(PlotSeries, initial=PlotSeries(", ".join(names), data=dataset))

        def sheet(names: list[str]) -> None:
            opened = open_sheet(dataset)
            if opened is not None:
                opened.show_picked(names)

        def select(names: list[str]) -> None:
            taken = set(current().objects)
            label = next(f"selection{k or ''}" for k in range(len(taken) + 2) if f"selection{k or ''}" not in taken)
            run_dialog(SelectSeries, initial=SelectSeries(", ".join(names), data=dataset, name=label))

        def generate(names: list[str]) -> None:
            run_dialog(GenerateSeries, initial=GenerateSeries("{x}_g", "", data=dataset, for_each=" ".join(names)))

        def describe(names: list[str]) -> None:
            from ..commands import DescriptiveStatistics

            run_dialog(DescriptiveStatistics, initial=DescriptiveStatistics(data=dataset, series=", ".join(names)))

        return {"plot": plot, "sheet": sheet, "select": select, "generate": generate, "describe": describe,
                "working_set": lambda names: apply_working_set(dataset, ", ".join(names))}

    def run_matrix_step(cmd: Any) -> bool:
        """Run a step a matrix menu made, then show the new matrix (or say why it could not be made)."""
        if busy():
            Qt.QMessageBox.information(window, "A step is running", "Wait for the running step to finish first.")
            return False
        try:
            current().run(cmd)
        except Exception as exc:
            refresh()
            Qt.QMessageBox.warning(window, *titled("Could not make the matrix", str(exc)))
            return False
        refresh()
        if cmd.name in current().objects:
            show_object(cmd.name)
        return True

    def matrix_menu_for(name: str) -> Any:
        """The Matrix menu's maker for a matrix's or a decomposition's window (None for other objects)."""
        from ..matrix import Matrix
        from ..matrix_expr import DECOMPOSITIONS

        if not isinstance(current().objects.get(name), (Matrix, *DECOMPOSITIONS)):
            return None
        from .matrix_qt import menu_for

        return lambda parent, n=name: menu_for(parent, n, current().objects[n], current(), run_matrix_step)

    def open_matrix_grid(initial: Any = None) -> Any:
        """Model > Matrix > Enter values: the grid; OK runs the step and opens the matrix."""
        from .matrix_qt import MatrixGridDialog

        dialog = MatrixGridDialog(current(), window, initial if type(initial).__name__ == "EnterMatrix" else None)
        window.matrix_grid = dialog
        if dialog.exec() and dialog.ran is not None:
            refresh()
            if dialog.ran.name in current().objects:
                show_object(dialog.ran.name)
        return dialog

    def next_steps_for(name: str) -> list[tuple[str, Any]]:
        """A result's Next menu: each step opens its dialog with the result filled in, in that stage's workspace;
        below a separator, the task pages ("How do I…") that carry on from that kind of result open in the help."""
        from .next_steps import pages_for, prefilled, steps_for

        def go(cls: type, field: str, stage: str) -> None:
            if getattr(window, "workspace", "all") != "all":  # in All, stay there
                window.switch_workspace(stage)
            initial = prefilled(cls, **{field: name})
            run_dialog(cls, initial=initial) if initial is not None else run_dialog(cls)

        obj = current()[name]
        steps: list[tuple[Any, Any]] = [(label, lambda c=cls, f=field, w=ws: go(c, f, w)) for label, cls, field, ws in steps_for(obj)]
        pages = [(f"How do I… {title}", lambda u=url: window.open_help(u)) for title, url in pages_for(obj)]
        return steps + ([(None, None)] if steps and pages else []) + pages

    def add_to_report(source: str, tab: str | None) -> None:
        """Put a result's tab (or, without one, the result) in the report outline, and say so."""
        try:
            current().add_to_report(source, tab)
        except Exception as exc:
            Qt.QMessageBox.warning(window, *titled("Add to report", str(exc)))
            return
        report_view.refresh(len(current().report_outline) - 1)
        update_title()
        count = sum(1 for i in current().report_outline if i["kind"] == "item")
        window.statusBar().showMessage(f"Added {source}{f' — {tab}' if tab else ''} to the report ({count} item"
                                       f"{'s' if count != 1 else ''}; the Report panel, in the Report workspace)", 6000)

    def export_report_outline() -> None:
        from ..commands import ExportReport

        run_dialog(ExportReport, initial=ExportReport.from_outline(current()))

    def apply_working_set(dataset: str, spec: str | None) -> bool:
        """Set (or clear) a dataset's working set, and show it in the windows open on that dataset."""
        try:
            current().set_working_set(dataset, spec)
        except Exception as exc:
            Qt.QMessageBox.warning(window, *titled("Working set", str(exc)))
            return False
        if dataset in sheets:
            sheets[dataset].apply_scope()
        if dataset in open_windows:
            show_object(dataset, activate=False)
        update_title()
        return True

    def choose_working_set(dataset: str) -> None:
        """Ask for the series to show: patterns, or one of the selections."""
        from ..selection import Selection

        s = current()
        choices = [s.working_sets.get(dataset, "")] + [n for n, o in s.objects.items() if isinstance(o, Selection)]
        spec, ok = Qt.QInputDialog.getItem(
            window, "Working set", f"The series of {dataset} to show in dialogs, the spreadsheet and its window:\n"
            "names and patterns (gdp_*, cpi*, u), or a selection. Empty: every series.", list(dict.fromkeys(choices)), 0, True)
        if ok:
            apply_working_set(dataset, spec.strip() or None)

    def use_selection_as_working_set(selection: str) -> None:
        from ..data import Dataset

        datasets = [n for n, o in current().objects.items() if isinstance(o, Dataset)]
        if not datasets:
            Qt.QMessageBox.information(window, "Working set", "There is no dataset yet.")
            return
        dataset = datasets[0] if len(datasets) == 1 else None
        if dataset is None:
            dataset, ok = Qt.QInputDialog.getItem(window, "Working set", f"Show {selection}'s series in:", datasets, 0, False)
            if not ok:
                return
        apply_working_set(dataset, selection)

    def objects_menu(pos: Any) -> None:
        from ..data import Dataset

        item = objects.itemAt(pos)
        if item is None:
            return
        menu = Qt.QMenu(objects)
        from ..selection import Selection

        menu.addAction("Open", lambda: show_object(item.text(0)))
        if isinstance(current().objects.get(item.text(0)), Dataset):
            menu.addAction("Edit as spreadsheet", lambda: open_sheet(item.text(0)))
            menu.addAction("Working set…", lambda: choose_working_set(item.text(0)))
            if item.text(0) in current().working_sets:
                menu.addAction("Clear working set", lambda: apply_working_set(item.text(0), None))
        if isinstance(current().objects.get(item.text(0)), Selection):
            menu.addAction("Use as working set…", lambda: use_selection_as_working_set(item.text(0)))
        if item.text(0) in current().objects:  # a fit's coefficients, covariance (a VAR's lag matrices) as matrices
            from .matrix_qt import result_matrix_menu

            as_matrix = result_matrix_menu(menu, item.text(0), current().objects[item.text(0)], current(), run_matrix_step)
            if as_matrix is not None:
                menu.addMenu(as_matrix)
        name = item.text(0)
        s = current()
        if name in s.pending():
            menu.addAction("Compute", lambda: compute([name]))
        menu.addAction("Recalculate", lambda: recompute([name])).setEnabled(name in s.stale() or name in s.pending())
        if name in s.objects:
            frozen = name in s.frozen
            menu.addAction("Unfreeze" if frozen else "Freeze", lambda: freeze(name, not frozen))
        record = s.producer(name)
        menu.addSeparator()
        show_step = menu.addAction("Show the step in History", lambda: history.select_record(record))
        copy_step = menu.addAction("Copy the step's code", lambda: Qt.QApplication.clipboard().setText(record.code if record else ""))
        show_step.setEnabled(record is not None)
        copy_step.setEnabled(record is not None)
        window.last_objects_menu = [a.text() for a in menu.actions() if a.text()]
        window.last_objects_qmenu = menu
        menu.popup(objects.viewport().mapToGlobal(pos))  # not exec(): nothing waits on the menu

    objects.setContextMenuPolicy(QtNS.ContextMenuPolicy.CustomContextMenu)
    objects.customContextMenuRequested.connect(objects_menu)

    queue: list[tuple[str, int]] = []  # statements of a Code-window run still to go

    def after_code(record: Any) -> None:
        """A cell has run in the Immediate window: refresh, point a Code window at its error line,
        and run the next statement of the script (stopping at the first failure)."""
        editor, first = running.pop("editor", None), running.pop("first", 1)
        if editor is not None:
            line = None if record.ok or immediate is None else error_line(immediate.last_error)
            editor.editor.mark_error(first + line - 1 if line else None)
            if not record.ok:
                queue.clear()
            elif queue:
                from PySide6.QtCore import QTimer

                QTimer.singleShot(0, lambda e=editor: next_statement(e))  # after this cell has finished
        refresh()

    def next_statement(editor: Any) -> None:
        if not queue or immediate is None:
            return
        code, line = queue.pop(0)
        running.update(editor=editor, first=line)
        immediate.execute(code, "editor")

    def send_immediate(code: str, source: str = "immediate") -> None:
        if immediate is None:
            try:
                current().run_code(code, source)
            except Exception as exc:
                Qt.QMessageBox.critical(window, "Error", f"{type(exc).__name__}: {exc}")
            refresh()
            return
        immediate_dock.show()
        immediate_dock.raise_()
        immediate.execute(code, source)

    def run_from_editor(code: str, first: int, editor: Any) -> None:
        """Run code from a Code window one top-level statement at a time: each is a step in the
        history (a loop or function definition is one statement); the first failure stops the run."""
        from .editor_qt import statements

        parts = statements(code, first)
        if not parts:
            return
        if immediate is None:  # no console: run directly and report errors here
            for text, line in parts:
                try:
                    current().run_code(text, "editor")
                    editor.editor.mark_error(None)
                except Exception as exc:
                    where = error_line(exc)
                    editor.editor.mark_error(line + where - 1 if where else line)
                    Qt.QMessageBox.critical(window, "Error", f"{type(exc).__name__}: {exc}")
                    break
            refresh()
            return
        queue[:] = parts
        immediate_dock.show()
        immediate_dock.raise_()
        next_statement(editor)

    def editors() -> list[Any]:
        return [sub.widget() for sub in mdi.subWindowList() if isinstance(sub.widget(), EditorWindow)]

    def open_editor(path: str | None = None, text: str = "") -> Any:
        editor = EditorWindow(run_from_editor, lambda: current().namespace, path=path, text=text)

        class EditorSub(Qt.QMdiSubWindow):
            def closeEvent(self, event: Any) -> None:
                if editor.confirm_close():
                    super().closeEvent(event)
                else:
                    event.ignore()

        sub = EditorSub()
        sub.setWidget(editor)
        sub.setAttribute(QtNS.WidgetAttribute.WA_DeleteOnClose)
        mdi.addSubWindow(sub)
        sub.resize(720, 520)
        editor.update_title()
        sub.show()
        fit_in_workspace(sub)
        editor.editor.setFocus()
        return editor

    def active_editor() -> Any:
        sub = mdi.activeSubWindow()
        return sub.widget() if sub is not None and isinstance(sub.widget(), EditorWindow) else None

    def send_editor(code: str) -> None:
        editor = active_editor() or (editors()[0] if editors() else None)
        if editor is None:  # a new script: start it with the imports a standalone script needs
            from ..commands.session import SCRIPT_HEADER

            open_editor(text=SCRIPT_HEADER + "\n" + code + "\n")
            return
        mdi.setActiveSubWindow(editor.parentWidget())
        editor.editor.insertPlainText(code if code.endswith("\n") else code + "\n")
        editor.editor.setFocus()

    def rerun(records: list[Any]) -> None:
        try:
            for record in records:
                current().run(record.command)
        except Exception as exc:
            Qt.QMessageBox.critical(window, *titled("Re-run stopped", str(exc)))
        refresh()

    def reopen(record: Any) -> None:
        run_dialog(type(record.command), initial=record.command)

    def table_heatmap(name: str, tab: str) -> None:
        """A result's table as a heatmap: the step runs at once (nothing to choose), and the figure is shown."""
        from ..commands import TableHeatmap

        session = current()
        try:
            session.run(TableHeatmap(name, tab, scale="auto"))  # each column its own scale where the units differ
        except Exception as exc:
            Qt.QMessageBox.critical(window, *titled("Could not make the heatmap", str(exc)))
            return
        refresh()
        show_object(session.history[-1].command._target())

    def place_note(name: str, tab: str, figure: Any, panel: int, x: Any, y: float) -> None:
        """Add a note where the figure was clicked: the dialog opens with the panel, date and value filled in."""
        from ..commands import AnnotateFigure
        from ..figures import Figure

        session = current()
        steps = len(session.history)
        run_dialog(AnnotateFigure, initial=AnnotateFigure.at(name, figure, tab, is_figure=isinstance(session[name], Figure),
                                                             panel=panel, x=x, y=y))
        if len(session.history) > steps and session.history[-1].ok:
            for written in session.history[-1].writes:
                if written in session.objects:
                    show_object(written)

    def figure_edit(name: str, tab: str, figure: Any, action: str, *args: Any) -> None:
        """An edit made on a figure itself. An arrow drawn on it is added at once, as an Add a note step, and selected
        to adjust; a change marker's two points open Mark a change, filled in; a drag, new words or Delete run as a
        Change a style or Recorded styles step at once (undone and redone as any step); a double-click opens Change a
        style on that step; "style" (series, panel) opens Figure style on that series or panel (the right-click menu's
        Format series… and Format axes…). The new figure is shown. On a result's graph, what is added goes on the
        figure already made from it (``<result>_graph``), so a second arrow joins the first."""
        import dataclasses

        from ..commands import AnnotateFigure, ArrangeStyles, ChangeStyle, MarkChange, StyleFigure
        from ..figures import Figure

        session = current()
        is_figure = isinstance(session[name], Figure)
        target = name if is_figure else f"{name}_{'_'.join(tab.lower().split())}"
        if action == "pick" and not is_figure and isinstance(session.objects.get(target), Figure):
            name, tab, figure, is_figure = target, "Graph", session[target], True  # add to what was added before
        steps = len(session.history)
        if action == "format":
            (step,) = args
            run_dialog(ChangeStyle, initial=ChangeStyle.for_step(name, figure, step, tab, is_figure))
        elif action == "style":
            series, panel = args
            run_dialog(StyleFigure, initial=dataclasses.replace(StyleFigure.for_figure(name, figure, tab, is_figure=is_figure),
                                                                series=series, panel=str(panel if panel is not None else 0)))
        elif action == "pick":
            kind, panel, first, second, series = args
            if kind == "arrow":
                command = AnnotateFigure.at(name, figure, tab, is_figure=is_figure, panel=panel, x=second[0], y=second[1],
                                            what="arrow", start=first)
                try:
                    session.run(command)
                except Exception as exc:  # noqa: BLE001 - shown, as any step's error
                    Qt.QMessageBox.critical(window, *titled("Could not add the arrow", str(exc)))
                    return
                refresh()
                made = command._target()
                show_object(made)
                select_style(made, len(session[made].style) - 1)
                return
            else:
                run_dialog(MarkChange, initial=MarkChange.at(name, figure, tab, is_figure=is_figure, series=series,
                                                             start=first[0], end=second[0]))
        else:
            if action == "restyle":
                step, props = args
                command: Any = ChangeStyle(name, tab, step=str(step), properties=", ".join(f"{k}={v!r}" for k, v in props.items()),
                                           name=target)
            else:
                (step,) = args
                command = dataclasses.replace(ArrangeStyles.for_figure(name, figure, tab, is_figure=is_figure),
                                              keep=", ".join(str(k) for k in range(len(figure.style)) if k != step))
            try:
                session.run(command)
            except Exception as exc:
                Qt.QMessageBox.critical(window, *titled("Could not change the figure", str(exc)))
                return
            refresh()
        if len(session.history) > steps and session.history[-1].ok:
            for written in session.history[-1].writes:
                if written in session.objects:
                    show_object(written)

    window.figure_edit = figure_edit

    def select_style(name: str, step: int) -> None:
        """Select a recorded step on the figure's open window, its handles shown (as after a click on it)."""
        sub = open_windows.get(name)
        tabs = None if sub is None else sub.widget()
        page = None if tabs is None or not hasattr(tabs, "currentWidget") else tabs.currentWidget()
        editor = getattr(page, "editor", None)
        if editor is not None:
            page.canvas.draw()  # the arrow's place on screen, to pick and drag it
            try:
                editor.select(step)
            except (KeyError, StopIteration):
                pass

    window.select_style = select_style

    def edit_figure(name: str, tab: str, figure: Any, kind: str = "options") -> None:
        """A figure's Edit…. Style opens Figure style (matplotlib's lines, axes, legend and size). Titles, shading and
        marks reopens a Plot series figure's dialog, as made, and opens Figure options, filled in with how it is drawn
        now, for any other figure (a result's too). The new figure is shown."""
        from ..commands import (AnnotateFigure, ArrangeStyles, CallAxesMethod, ChangeStyle, CopyStyle, FigureOptions, PlotSeries,
                                SaveFigure, SetProperties, StyleFigure, UseHouseStyle)
        from ..figures import Figure

        session = current()
        obj = session[name]
        record = session.producer(name)
        steps = len(session.history)
        is_figure = isinstance(obj, Figure)
        if kind == "styles":
            import dataclasses

            if not figure.style:
                Qt.QMessageBox.information(window, "Recorded styles", "No styles are recorded on this figure yet: "
                                           "Edit… → Style, All properties, Add a note or matplotlib method record them.")
                return
            dialog = style_steps_dialog(figure, window)
            if not dialog.exec():
                return
            order, target = dialog.order(), name
            if order != list(range(len(figure.style))):  # removed or moved: one step
                command = ArrangeStyles.for_figure(name, figure, tab, is_figure=is_figure)
                command = dataclasses.replace(command, keep=", ".join(str(k) for k in order))
                try:
                    session.run(command)
                except Exception as exc:
                    Qt.QMessageBox.critical(window, *titled("Could not change the styles", str(exc)))
                    return
                refresh()
                target = command._target()
                figure, tab, is_figure = session[target], "Graph", True
            if dialog.change is not None and dialog.change in order:
                run_dialog(ChangeStyle, initial=ChangeStyle.for_step(target, figure, order.index(dialog.change), tab, is_figure))
        elif kind == "house":
            run_dialog(UseHouseStyle, initial=UseHouseStyle(name, tab))
        elif kind == "save":
            run_dialog(SaveFigure, initial=SaveFigure.for_figure(name, figure, tab, is_figure=is_figure))
        elif kind == "copy":
            run_dialog(CopyStyle, initial=CopyStyle.for_figure(name, figure, tab, is_figure=is_figure))
        elif kind == "style":
            run_dialog(StyleFigure, initial=StyleFigure.for_figure(name, figure, tab, is_figure=isinstance(obj, Figure)))
        elif kind == "properties":
            run_dialog(SetProperties, initial=SetProperties.for_figure(name, figure, tab, is_figure=isinstance(obj, Figure)))
        elif kind == "method":
            run_dialog(CallAxesMethod, initial=CallAxesMethod.for_figure(name, figure, tab, is_figure=isinstance(obj, Figure)))
        elif kind == "note":
            run_dialog(AnnotateFigure, initial=AnnotateFigure.at(name, figure, tab, is_figure=isinstance(obj, Figure)))
        elif kind in ("arrow", "textbox"):
            run_dialog(AnnotateFigure, initial=AnnotateFigure.at(name, figure, tab, is_figure=isinstance(obj, Figure), what=kind))
        elif kind == "change":
            from ..commands import MarkChange

            series = next((getattr(layer, "label", "") for p in figure.panels for layer in p.layers if getattr(layer, "label", "")), "")
            run_dialog(MarkChange, initial=MarkChange.at(name, figure, tab, is_figure=isinstance(obj, Figure), series=series))
        elif isinstance(obj, Figure) and record is not None and isinstance(record.command, PlotSeries):
            run_dialog(PlotSeries, initial=record.command)
        else:
            run_dialog(FigureOptions, initial=FigureOptions.for_figure(name, figure, tab, is_figure=isinstance(obj, Figure)))
        if len(session.history) > steps and session.history[-1].ok:  # not cancelled: show what it made
            for written in session.history[-1].writes:
                if written in session.objects:
                    show_object(written)

    def new_script_from_session() -> None:
        open_editor(text=current().script())

    def open_script() -> None:
        path, _ = Qt.QFileDialog.getOpenFileName(window, "Open script", "", "Python (*.py);;All files (*)")
        if path:
            open_editor(path=path)

    def focused_editor() -> Any:
        widget = Qt.QApplication.focusWidget()
        while widget is not None and not isinstance(widget, EditorWindow):
            widget = widget.parentWidget()
        return widget

    # projects

    def confirm_discard() -> bool:
        if not current().dirty:
            return True
        answer = Qt.QMessageBox.question(window, "Unsaved work", "Save the current project first?",
                                         Qt.QMessageBox.StandardButton.Save | Qt.QMessageBox.StandardButton.Discard | Qt.QMessageBox.StandardButton.Cancel)
        if answer == Qt.QMessageBox.StandardButton.Save:
            return save(as_new=False)
        return answer == Qt.QMessageBox.StandardButton.Discard

    def close_start_page() -> None:
        page, window.start_page = getattr(window, "start_page", None), None
        if page is not None:
            try:
                page.close()
            except RuntimeError:  # already closed (its own buttons close it)
                pass

    def switch(new: Session) -> None:
        from .help_qt import forget_project_courses

        close_start_page()  # another session: the old one's Start window goes
        for sub in list(open_windows.values()):  # object windows; Code windows stay open
            sub.close()
        for sheet in list(sheets.values()):  # spreadsheets belong to the old session's datasets
            sheet.model.pending.clear()  # the user has already chosen to keep or discard the session's work
            sheet.parentWidget().close()
        forget_project_courses(state["session"], settings)  # the new project's courses were loaded when it was opened
        forget_autosave()  # the old session's copy goes with it (its changes were saved or discarded)
        state["session"] = new
        if immediate is not None:
            immediate.bind(new)
        refresh()

    def new_session() -> None:
        if confirm_discard():
            switch(Session())
            window.start_page = start_page(window, run_dialog, open_project, open_help, recent_projects(), open_demo)  # where to begin, again

    def built_demo(name: str) -> Session:
        """A demo built step by step in the background, as a long step runs: the window stays live, and after a moment
        the status bar shows progress and Cancel (nk_estimation estimates a model for ten seconds or more, and a window
        that freezes without a word looks broken). Cancelled, it raises ``Cancelled`` and nothing changes."""
        from PySide6 import QtCore

        from ..demos import demo

        window.statusBar().showMessage(f"Building demo {name}: each step runs, as in the History…")
        Qt.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.BusyCursor)
        Qt.QApplication.processEvents()
        try:
            return run_blocking(lambda: demo(name), f"demo {name}")
        finally:
            Qt.QApplication.restoreOverrideCursor()
            window.statusBar().clearMessage()

    def open_built_demo(name: str) -> None:
        """Build the demo (unless a step is running) and open it, after asking to save unsaved work."""
        from ..progress import Cancelled

        if busy():
            window.statusBar().showMessage("A step is running: wait for it, or Cancel it, before opening a demo", 10000)
            return
        if not confirm_discard():
            return
        try:
            built = built_demo(name)
        except Cancelled:
            window.statusBar().showMessage(f"Demo {name} not opened: cancelled", 10000)
            return
        switch(built)
        show_demo(name)

    def open_demo(name: str | None = None) -> None:
        """Open a demo project (asking which without ``name``), after asking to save unsaved work; also a panel's
        ``app.open_demo(name)`` (extension API 1.7). An unknown name is a ``NotFoundError`` that lists the demos."""
        from ..demos import demos
        from ..errors import NotFoundError

        names = demos()
        if name is None:  # a list of the demos, each with what it shows and opens (pymetrix.ui.demo_chooser)
            from . import demo_chooser

            name = demo_chooser.choose_demo(window)
            if name is None:
                return
        if name not in names:  # before asking about unsaved work
            raise NotFoundError(f"no demo {name!r}; demos: {', '.join(names)}")
        open_built_demo(name)

    def show_demo(name: str) -> None:
        """A demo opened: its headline results in windows, each on the view that shows the work, the first selected
        in Objects, and a note of what is open and where to go next."""
        from ..demos import headline

        windows, note = headline(name)
        shown = []
        for obj, view in windows:
            if obj in current().objects:
                show_object(obj)
                sub = open_windows.get(obj)
                if sub is not None and hasattr(sub.widget(), "select"):
                    sub.widget().select(view)
                shown.append(obj)
        if shown:
            top = open_windows.get(shown[-1])
            area = mdi.viewport().rect()
            if top is not None and area.width() > 200 and area.height() > 200:  # the headline: room for its figure or table
                top.resize(max(top.width(), int(area.width() * 0.85)), max(top.height(), int(area.height() * 0.85)))
                top.move(max(0, min(top.x(), area.width() - top.width())), max(0, min(top.y(), area.height() - top.height())))
            found = objects.findItems(shown[-1], QtNS.MatchFlag.MatchExactly, 0)
            if found:
                objects.setCurrentItem(found[0])
        window.statusBar().showMessage(f"Demo {name}: {note}. History shows each step." if note else
                                       "Demo opened: History and Script show each step", 20000)

    window.show_demo = show_demo
    app_interface._open_demo = open_demo  # panels open demos as File → Open demo project does

    def reload_demo() -> None:
        """Open the current demo again as it first was (after trying things in it during a walkthrough)."""
        name = getattr(current(), "demo", None)
        if not name:
            window.statusBar().showMessage("This is not a demo project: File → Open demo project… opens one", 5000)
            return
        open_built_demo(name)

    window.reload_demo = reload_demo

    _RECENT = 8  # projects listed under File > Open recent

    def recent_projects() -> list[str]:
        """Projects opened or saved here, newest first, those that still exist."""
        listed = settings.value("recent_projects", []) or []
        listed = [listed] if isinstance(listed, str) else list(listed)
        return [p for p in listed if Path(p).is_file()]

    def remember(path: str) -> None:
        resolved = str(Path(path).resolve())
        settings.setValue("recent_projects", [resolved, *[p for p in recent_projects() if p != resolved]][:_RECENT])

    def open_project(path: str | None = None) -> None:
        if not confirm_discard():
            return
        if path is None:
            path, _ = Qt.QFileDialog.getOpenFileName(window, "Open project", "", "PyMetrix projects (*.pymx)")
        if path:
            try:
                from ..commands.project import project_code

                info = project_code(path)
                trust = None
                if info.needs_trust:
                    trust = ask_trust(window, Path(path).name, info)
                    if trust is None:
                        return
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    switch(load_project(path, trust=trust))
                remember(path)
                if current().missing_plugins:  # its steps are kept, not run: say which plugins they need
                    from .plugins_qt import missing_plugins_note

                    Qt.QMessageBox.information(window, "Plugins needed", missing_plugins_note(current().missing_plugins))
            except Exception as exc:
                Qt.QMessageBox.critical(window, *titled("Could not open the project", str(exc)))

    def save(as_new: bool) -> bool:
        s = current()
        path = s.path
        if as_new or path is None:
            path, _ = Qt.QFileDialog.getSaveFileName(window, "Save project", str(path or "project.pymx"), "PyMetrix projects (*.pymx)")
            if not path:
                return False
        try:
            save_project(s, path)
        except Exception as exc:
            Qt.QMessageBox.critical(window, *titled("Could not save the project", str(exc)))
            return False
        remember(path)
        update_title()
        forget_autosave()  # saved: the copy for recovery is no longer needed
        return True

    def chosen_object() -> str | None:
        """The object of the active window, or else the one selected in the Objects pane."""
        sub = mdi.activeSubWindow()
        for name, shown in open_windows.items():
            if shown is sub:
                return name
        item = objects.currentItem()
        return item.text(0) if item is not None and item.text(0) in current().objects else None

    def save_result_file(path: str | None = None) -> None:
        """Save one result to a .pymxr file (pymetrix.saved), to send or keep outside the project."""
        from ..data import Dataset
        from ..saved import save_result

        name = chosen_object()
        if name is None:
            Qt.QMessageBox.information(window, "Save result", "Open a result's window, or select it in the Objects pane, then save it.")
            return
        obj = current().objects[name]
        if isinstance(obj, Dataset):
            Qt.QMessageBox.information(window, "Save result", f"{name} is a dataset: it is saved with the project, or as a data file "
                                       f"in code ({name}.write(\"{name}.parquet\")).")
            return
        if path is None:
            path, _ = Qt.QFileDialog.getSaveFileName(window, "Save result", f"{name}.pymxr", "PyMetrix results (*.pymxr)")
        if path:
            try:
                written = save_result(obj, path, name=name)
            except Exception as exc:
                Qt.QMessageBox.critical(window, *titled("Could not save the result", str(exc)))
                return
            window.statusBar().showMessage(f"{name} saved to {written.name}: File → Open result… reads it back", 8000)

    window.save_result = save_result_file

    def export_script() -> None:
        path, _ = Qt.QFileDialog.getSaveFileName(window, "Export script", "session.py", "Python (*.py)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(current().script())

    from ..plugins import load, set_mode
    from .plugins_qt import ask_new_plugins

    set_mode("app")  # a new plugin waits for an answer before any of it runs
    ask_new_plugins(window)
    load("commands")
    from ..commands.menus import MENU_TOPS, menu_entries

    menus: dict[str, Any] = {}
    tops = [i.menu.partition("/")[0] for i in COMMANDS.values() if i.menu]
    taken = ""
    for top in [t for t in MENU_TOPS if t in tops] + [t for t in dict.fromkeys(tops) if t not in MENU_TOPS]:
        title = with_access_key(top, taken)
        taken += title[title.index("&") + 1] if "&" in title else ""  # two plugins' menus never share a letter
        menus[top] = owned_menu(window.menuBar(), title)
    from PySide6.QtGui import QKeySequence

    def save_focused(as_new: bool) -> None:
        """Ctrl+S saves the script being edited, otherwise the project."""
        editor = focused_editor()
        if editor is not None:
            editor.save(as_new=as_new)
        else:
            save(as_new=as_new)

    file_menu = menus["File"]
    for label, handler, keys in (("New session", new_session, QKeySequence.StandardKey.New),
                                 ("Open project…", open_project, QKeySequence.StandardKey.Open),
                                 ("Open demo project…", open_demo, None),
                                 ("Reload demo project", reload_demo, None),
                                 ("Save project", lambda: save_focused(as_new=False), QKeySequence.StandardKey.Save),
                                 ("Save project as…", lambda: save_focused(as_new=True), QKeySequence.StandardKey.SaveAs),
                                 ("Export script…", export_script, None),
                                 ("Save result…", save_result_file, None),
                                 (None, None, None),
                                 ("New script", lambda: open_editor(), "Ctrl+Shift+N"),
                                 ("New script from session", new_script_from_session, None),
                                 ("Open script…", open_script, "Ctrl+Shift+O")):
        if label is None:
            file_menu.addSeparator()
            continue
        action = file_menu.addAction(label)
        action.triggered.connect(lambda _=False, h=handler: h())
        if keys is not None:
            action.setShortcut(QKeySequence(keys))
        if label == "Open project…":  # the projects opened or saved here, newest first
            recent_menu = owned_menu(file_menu, "Open recent")

            def list_recent(menu: Any = recent_menu) -> None:
                menu.clear()
                found = recent_projects()
                for number, path in enumerate(found, 1):
                    item = menu.addAction(f"&{number} {Path(path).name}" if number < 10 else Path(path).name)
                    item.setToolTip(path)
                    item.setStatusTip(path)
                    item.triggered.connect(lambda _=False, p=path: open_project(p))
                if not found:
                    menu.addAction("No recent projects").setEnabled(False)
                menu.addSeparator()
                clear = menu.addAction("Clear recent")
                clear.setEnabled(bool(found))
                clear.triggered.connect(lambda: settings.setValue("recent_projects", []))

            recent_menu.aboutToShow.connect(list_recent)
            window.recent_menu, window.list_recent = recent_menu, list_recent
    file_menu.addSeparator()

    # Edit, after File: Undo takes back the last dialog step
    edit_menu = Qt.QMenu(with_access_key("Edit"), window.menuBar())
    _MENUS.append(edit_menu)
    bar_actions = window.menuBar().actions()
    following = bar_actions.index(file_menu.menuAction()) + 1
    if following < len(bar_actions):
        window.menuBar().insertMenu(bar_actions[following], edit_menu)
    else:
        window.menuBar().addMenu(edit_menu)
    undo_action = edit_menu.addAction("Undo")
    undo_keys = QKeySequence(QKeySequence.StandardKey.Undo)
    undo_action.setShortcut(undo_keys)
    redo_action = edit_menu.addAction("Redo")
    # Ctrl+Y (Windows') and Ctrl+Shift+Z (elsewhere), both everywhere; not the platform's other bindings (Windows' Alt+
    # Shift+Backspace, a keyboard's Redo key), which made Keyboard shortcuts read "Ctrl+Y, Alt+Shift+Backspace, ..."
    redo_keys = [QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")]
    redo_action.setShortcuts(redo_keys)

    def update_undo() -> None:
        label = current().undo_label()
        undo_action.setText(f"Undo {label}" if label else "Undo")
        undo_action.setEnabled(bool(label) and not busy())
        undo_action.setToolTip(current().undo_blocked() or "take back the last step: the objects as they were before it")
        again = current().redo_label()
        redo_action.setText(f"Redo {again}" if again else "Redo")
        redo_action.setEnabled(bool(again) and not busy())
        redo_action.setToolTip(current().redo_blocked() or "run the step undone again, as it was")

    def redo_step() -> None:
        from ..commands.session import CommandError

        if busy():
            return
        try:
            record = current().redo()
        except CommandError as exc:
            window.statusBar().showMessage(f"Cannot redo: {exc}", 10000)
            return
        except Exception as exc:  # noqa: BLE001 - the step failed this time: say so, as for any step
            Qt.QMessageBox.critical(window, *titled("Redo stopped", str(exc)))
            refresh()
            return
        refresh()
        update_title()
        line = record.code.strip().splitlines()[0] if record.code.strip() else type(record.command).__name__
        window.statusBar().showMessage(f"Redid {line}", 10000)
        for written in record.writes:
            if written in current().objects and written in open_windows:
                show_object(written, activate=False)

    def undo_step() -> None:
        from ..commands.session import CommandError

        if busy():
            return
        losing = sorted(n for n in current().undo_changes() if n in sheets and sheets[n].model.pending)
        if losing:  # a spreadsheet of a dataset Undo takes away has edits not applied yet: they would go with it
            answer = Qt.QMessageBox.question(window, "Unapplied edits", f"The spreadsheet of {', '.join(losing)} has "
                                             "edits not applied yet; Undo closes it and they are lost. Undo anyway?")
            if answer != Qt.QMessageBox.StandardButton.Yes:
                return
        try:
            record = current().undo()
        except CommandError as exc:
            window.statusBar().showMessage(f"Cannot undo: {exc}", 10000)
            return
        refresh()
        update_title()
        line = record.code.strip().splitlines()[0] if record.code.strip() else type(record.command).__name__
        window.statusBar().showMessage(f"Undid {line}", 10000)

    watched_sheets: set[int] = set()

    def own_undo(widget: Any) -> bool:
        """Whether the widget with focus has an Undo of its own (text, a spreadsheet, a Code window)."""
        from .sheet_qt import SheetWindow

        while widget is not None:
            if isinstance(widget, (Qt.QLineEdit, Qt.QTextEdit, Qt.QPlainTextEdit, Qt.QAbstractSpinBox)):
                return not widget.isReadOnly()  # read-only text (the Script panel, help) has nothing to undo
            if isinstance(widget, SheetWindow):
                if id(widget) not in watched_sheets:  # after Apply its stack is empty: Ctrl+Z goes back to the steps
                    watched_sheets.add(id(widget))
                    widget.undo.canUndoChanged.connect(lambda _can: follow_focus(None, Qt.QApplication.focusWidget()))
                return widget.undo.canUndo()
            widget = widget.parentWidget()
        return False

    def follow_focus(_old: Any, new: Any) -> None:  # Ctrl+Z is the field's while it has its own Undo, else the step's
        try:
            own = own_undo(new)
            undo_action.setShortcut(QKeySequence() if own else undo_keys)
            redo_action.setShortcuts([] if own else redo_keys)
        except RuntimeError:  # the window is gone
            pass

    def stop_following(*_: Any) -> None:
        try:
            Qt.QApplication.instance().focusChanged.disconnect(follow_focus)
        except (RuntimeError, TypeError):  # already disconnected
            pass

    Qt.QApplication.instance().focusChanged.connect(follow_focus)
    window.destroyed.connect(stop_following)
    undo_action.triggered.connect(undo_step)
    redo_action.triggered.connect(redo_step)
    window.addAction(undo_action)  # their keys work in every workspace, Edit shown or not
    window.addAction(redo_action)
    edit_menu.aboutToShow.connect(update_undo)
    window.edit_menu, window.undo_action, window.update_undo, window.undo_step = edit_menu, undo_action, update_undo, undo_step
    window.redo_action, window.redo_step = redo_action, redo_step
    window.follow_focus = follow_focus

    def fill(menu: Any, entries: list[Any]) -> None:
        """A menu's commands as laid out (pymetrix.commands.menus): grouped, with separators and submenus."""
        for entry in entries:
            if entry.kind == "separator":
                menu.addSeparator()
            elif entry.kind == "submenu":  # owned by its menu and kept: older PySide6 otherwise deletes it
                submenu = Qt.QMenu(entry.label, menu)
                menu.addMenu(submenu)
                submenus.append(submenu)
                fill(submenu, entry.entries)
            elif entry.kind == "action" and entry.action == "spreadsheet":
                sheet_action = menu.addAction("Spreadsheet editor…")
                sheet_action.setShortcut(QKeySequence("Ctrl+E"))
                sheet_action.triggered.connect(lambda: open_sheet())
            elif entry.kind == "action" and entry.action == "series-browser":
                menu.addAction("Browse series…").triggered.connect(lambda: open_series_browser())
            elif entry.kind == "action" and entry.action == "refresh-data":
                menu.addAction("Refresh data…").triggered.connect(lambda: open_refresh_data())
            elif entry.kind == "command":
                menu.addAction(entry.label).triggered.connect(lambda _=False, c=entry.info.cls: run_dialog(c))

    infos = [i for i in COMMANDS.values() if i.menu]
    submenus: list[Any] = []
    for top, menu in menus.items():
        fill(menu, menu_entries(top, infos))
    window.command_menus, window.command_submenus = menus, submenus
    quit_separator = file_menu.addSeparator()  # last in File: leave the app (asking to save, as closing the window does)
    quit_action = file_menu.addAction("&Quit" if sys.platform == "darwin" else "E&xit")
    quit_action.setShortcut(QKeySequence("Ctrl+Q"))
    quit_action.setMenuRole(quit_action.MenuRole.QuitRole)
    quit_action.setToolTip("close pymetrix; unsaved work is offered for saving first")
    quit_action.triggered.connect(window.close)
    window.quit_action = quit_action

    def go_to_window() -> None:
        subs = mdi.subWindowList()
        if not subs:
            Qt.QMessageBox.information(window, "Go to window", "No windows are open. Double-click an object to open one.")
            return
        titles = [s.windowTitle() for s in subs]
        title, ok = Qt.QInputDialog.getItem(window, "Go to window", "Window:", titles, 0, False)
        if ok:
            target = subs[titles.index(title)]
            target.showNormal()
            mdi.setActiveSubWindow(target)

    window.go_to_window = go_to_window
    def zoom(step: int) -> None:
        """Zoom the active window only (see pymetrix.ui.zoom): the menus, docks and other windows keep their size."""
        sub = mdi.activeSubWindow()
        widget = sub.widget() if sub is not None else None
        if widget is None:
            window.statusBar().showMessage("No window to zoom: open an object first", 3000)
            return
        action = {1: "zoom_in", -1: "zoom_out", 0: "zoom_reset"}[step]
        if not hasattr(widget, action):
            window.statusBar().showMessage("This window does not zoom", 3000)
            return
        getattr(widget, action)()
        window.statusBar().showMessage(f"Zoom {round(100 * widget.zoom_factor())}%", 3000)

    window.zoom = zoom
    view = owned_menu(window.menuBar(), with_access_key("View"))
    def palette_entries() -> list[Any]:
        """Everything Find anything searches: each menu item under its path, the workspace's objects, the help pages."""
        from .. import help as hp
        from .palette import PaletteEntry

        out: list[Any] = []

        def walk(menu: Any, path: str) -> None:
            for action in menu.actions():
                if action.isSeparator() or not action.isVisible():
                    continue
                text = action.text().replace("&", "")
                if action.menu() is not None:
                    walk(action.menu(), f"{path} → {text}")
                elif text and action.isEnabled() and action is not palette_action:
                    out.append(PaletteEntry(f"{path} → {text}", "Menu", action.trigger))

        for top in window.menuBar().actions():
            if top.menu() is not None:
                walk(top.menu(), top.text().replace("&", ""))
        for name, obj in current().objects.items():
            out.append(PaletteEntry(name, "Object", lambda n=name: show_object(n), type(obj).__name__, icon=object_icon(obj)))
        for section, pages in hp.contents():
            out.extend(PaletteEntry(title, "Help", lambda u=url: window.open_help(u), section) for title, url in pages)
        from ..coming_from import palette_synonyms
        from .palette import synonym_entries

        # EViews', IRIS's and Dynare's names (genr, smpl, sstate): after what pymetrix's own names find
        return out + synonym_entries(out, palette_synonyms(), lambda key: window.open_help(hp.link("topic", key)))

    def find_anything() -> Any:
        """The command palette (Ctrl+K): type a few letters of anything, Enter opens it."""
        from .palette import palette_dialog

        dialog = palette_dialog(palette_entries, window)
        dialog.exec()
        return dialog

    palette_action = view.addAction("Find anything…")
    palette_action.setShortcut(QKeySequence("Ctrl+K"))
    palette_action.setToolTip("a command, an object or a help page, by typing a few letters of it")
    palette_action.triggered.connect(find_anything)
    window.palette_entries, window.find_anything = palette_entries, find_anything
    go = view.addAction("Go to window…")
    go.setShortcut(QKeySequence("Ctrl+G"))
    go.triggered.connect(go_to_window)

    def find_object() -> None:
        objects_dock.show()
        objects_dock.raise_()
        object_filter.setFocus()
        object_filter.selectAll()

    find = view.addAction("Find object…")
    find.setShortcut(QKeySequence("Ctrl+Shift+F"))  # Ctrl+F finds text in Code windows
    find.triggered.connect(find_object)
    view.addSeparator()
    view.addAction(objects_dock.toggleViewAction())
    view.addAction(history_dock.toggleViewAction())
    view.addAction(immediate_dock.toggleViewAction())
    view.addAction(script_dock.toggleViewAction())
    view.addAction(problems_dock.toggleViewAction())
    view.addAction(report_dock.toggleViewAction())
    view.addSeparator()
    view.addAction("Tile windows").triggered.connect(mdi.tileSubWindows)
    view.addAction("Cascade windows").triggered.connect(mdi.cascadeSubWindows)
    tabbed = view.addAction("Tabbed windows")
    tabbed.setCheckable(True)

    def window_menu(sub: Any, menu: Any = None) -> Any:
        """A window's own menu (its title bar's, or its tab's in tabbed view): copy it as an image, add its result to
        the report, close it or the others, tile or cascade them all."""
        menu = menu if menu is not None else Qt.QMenu(window)
        name = next((n for n, w in open_windows.items() if w is sub), None)
        menu.addAction("Copy the window as an image", lambda: Qt.QApplication.clipboard().setPixmap(sub.widget().grab()))
        if name is not None and name in current().objects:
            menu.addAction(f"Add “{name}” to the report", lambda: add_to_report(name, None))
        menu.addSeparator()
        menu.addAction("Close", sub.close)
        menu.addAction("Close the other windows", lambda: [w.close() for w in mdi.subWindowList() if w is not sub])
        menu.addAction("Tile windows", mdi.tileSubWindows)
        menu.addAction("Cascade windows", mdi.cascadeSubWindows)
        return menu

    def extend_window_menu(sub: Any) -> None:
        """Once per window: its title bar's menu (Qt's: restore, move, size, close) gains the window menu's actions."""
        if sub is None or getattr(sub, "pymetrix_menu", False):
            return
        sub.pymetrix_menu = True
        system = sub.systemMenu()
        if system is not None:
            system.addSeparator()
            system.aboutToShow.connect(lambda m=system, w=sub: refill_system_menu(m, w))

    def refill_system_menu(menu: Any, sub: Any) -> None:
        """The window menu's actions, as they are now (a result may have been added since), after Qt's own."""
        for action in getattr(menu, "pymetrix_actions", []):
            menu.removeAction(action)
        before = list(menu.actions())
        window_menu(sub, menu)
        menu.pymetrix_actions = [a for a in menu.actions() if a not in before]

    def tab_bar_menu(pos: Any) -> None:
        bar = mdi.findChild(Qt.QTabBar)
        index = bar.tabAt(pos) if bar is not None else -1
        subs = mdi.subWindowList()
        if 0 <= index < len(subs):
            window_menu(subs[index]).popup(bar.mapToGlobal(pos))

    def set_tabbed(on: bool) -> None:
        mdi.setViewMode(Qt.QMdiArea.ViewMode.TabbedView if on else Qt.QMdiArea.ViewMode.SubWindowView)
        bar = mdi.findChild(Qt.QTabBar)
        if on and bar is not None and not getattr(bar, "pymetrix_menu", False):
            bar.pymetrix_menu = True
            size_icons(bar)  # the windows' icons on their tabs at the text size
            bar.setContextMenuPolicy(QtNS.ContextMenuPolicy.CustomContextMenu)
            bar.customContextMenuRequested.connect(tab_bar_menu)

    mdi.subWindowActivated.connect(extend_window_menu)
    window.window_menu = window_menu
    tabbed.toggled.connect(set_tabbed)
    view.addSeparator()
    for label, step, keys in (("Zoom in", 1, ("Ctrl+=", "Ctrl++")), ("Zoom out", -1, ("Ctrl+-",)), ("Actual size", 0, ("Ctrl+0",))):
        action = view.addAction(label)
        action.setShortcuts([QKeySequence(k) for k in keys])
        action.setToolTip("the active window only")
        action.triggered.connect(lambda _=False, s=step: zoom(s))
    from PySide6.QtGui import QActionGroup

    sizes_menu = owned_menu(view, "Text size")  # the whole app's text, kept; the zoom above multiplies it, window by window
    sizes_menu.setToolTipsVisible(True)
    sizes = QActionGroup(sizes_menu)
    window.text_size_actions = {}
    for step in text_size.STEPS:
        action = sizes_menu.addAction(step)
        action.setCheckable(True)
        action.setToolTip(f"every window's text at {round(100 * text_size.FACTORS[step])}% of the system's; kept for next time")
        sizes.addAction(action)
        action.triggered.connect(lambda _=False, s=step: set_text_size(s))
        window.text_size_actions[step] = action
    window.text_size_actions[text_size.current()].setChecked(True)

    def set_text_size(step: str) -> None:
        """View → Text size: the whole app at ``step`` now (open windows drawn again, so their tables and charts take
        it), and kept for the next start."""
        text_size.apply(step)
        text_size.save(step)
        window.text_size_actions[step].setChecked(True)
        for sub in list(open_windows.values()):  # result windows made again: column widths, row heights, charts
            sub.shown = None
        for name in list(open_windows):
            if name in current().objects:
                show_object(name, activate=False)
        fit_object_columns()
        window.statusBar().showMessage(f"Text size: {step}", 3000)

    window.set_text_size = set_text_size
    view.addSeparator()
    view.addAction("Close all windows").triggered.connect(mdi.closeAllSubWindows)
    reset_layout = view.addAction("Reset layout")
    reset_layout.setToolTip("the panels of this workspace as it first shows them")
    reset_layout.triggered.connect(lambda: (window.restoreState(default_state), window.reset_workspace()))
    view.addSeparator()
    from ..notation import set_notation

    algebraic = view.addAction("Algebraic notation")  # equations as cₜ = β₀ + β₁ yₜ + ...; scripts keep the code form
    algebraic.setCheckable(True)
    algebraic.setToolTip("show equations and coefficients in algebraic notation (new windows); scripts and projects are unchanged")

    def toggle_notation(on: bool) -> None:
        set_notation("algebraic" if on else "code")
        settings.setValue("notation", "algebraic" if on else "code")

    algebraic.toggled.connect(toggle_notation)
    algebraic.setChecked(settings.value("notation", "code") == "algebraic")
    window.algebraic_action = algebraic
    view.addSeparator()
    window_actions: list[Any] = []

    def list_windows() -> None:  # the open windows, at the foot of the View menu
        for action in window_actions:
            view.removeAction(action)
        window_actions.clear()
        active = mdi.activeSubWindow()
        for sub in mdi.subWindowList():
            action = view.addAction(sub.windowTitle())
            action.setCheckable(True)
            action.setChecked(sub is active)
            action.triggered.connect(lambda _=False, s=sub: (s.showNormal(), mdi.setActiveSubWindow(s)))
            window_actions.append(action)

    view.aboutToShow.connect(list_windows)

    calculate = owned_menu(window.menuBar(), with_access_key("Calculate"))
    from PySide6.QtGui import QActionGroup

    modes = QActionGroup(calculate)
    automatic_action = calculate.addAction("Recalculate automatically")  # the pair says what it sets, not just a mode
    manual_action = calculate.addAction("Recalculate on request (F9)")
    for action, mode, tip in ((automatic_action, "automatic", "results whose inputs change are recomputed at once"),
                              (manual_action, "manual", "results whose inputs change are marked out of date until you recalculate")):
        action.setCheckable(True)
        action.setToolTip(tip)
        modes.addAction(action)
        action.triggered.connect(lambda _=False, m=mode: (current().set_mode(m), refresh()))
    calculate.addSeparator()
    recalc_action = calculate.addAction("Recalculate")
    recalc_action.setShortcut(QKeySequence("F9"))
    recalc_action.triggered.connect(lambda: recompute())
    calculate.addAction("Compute results not computed").triggered.connect(lambda: compute())
    calculate.addSeparator()
    calculate.addAction("Clean up history…").triggered.connect(clean_up_history)
    calculate.addSeparator()  # the generator the next simulations draw with: a setting of the calculation
    from .._random import GENERATORS, get_random_generator, set_random_generator

    generators = owned_menu(calculate, "Random generator")  # the bit generator every simulation, bootstrap and sampler draws with
    generators.setToolTipsVisible(True)
    generator_group = QActionGroup(generators)
    labels = {"pcg64": "PCG64 (default)", "mt19937": "Mersenne Twister (MT19937)", "philox": "Philox", "sfc64": "SFC64"}
    saved_generator = str(settings.value("random_generator", "")) if settings.contains("random_generator") else ""
    if saved_generator in GENERATORS:  # the choice made in an earlier session
        set_random_generator(saved_generator)
    window.generator_actions = {}
    for key in GENERATORS:
        action = generators.addAction(labels[key])
        action.setCheckable(True)
        action.setActionGroup(generator_group)
        action.setChecked(key == get_random_generator())
        action.setToolTip("used for new steps, and saved with each step of the project; even the Mersenne Twister does not "
                          "reproduce MATLAB's or Dynare's draws")

        def choose(on: bool, k: str = key) -> None:
            if on:
                set_random_generator(k)
                settings.setValue("random_generator", k)

        action.toggled.connect(choose)
        window.generator_actions[key] = action
    calc_mode.update = lambda s: (automatic_action.setChecked(s.mode == "automatic"), manual_action.setChecked(s.mode == "manual"))
    calc_mode.update(current())

    help_menu = owned_menu(window.menuBar(), with_access_key("Help"))

    def show_plugins() -> None:
        from .plugins_qt import plugin_manager

        plugin_manager(window).exec()

    from .help_qt import course_dialog, help_window, restore_courses

    help_windows: list[Any] = []

    def open_help(start: str | None = None) -> Any:
        if help_windows and help_windows[0].isVisible():
            target = help_windows[0]
            target.refresh()
            if start:
                target.view.go(start)
        else:
            help_windows[:] = [help_window(window, start)]
            target = help_windows[0]
        target.show()
        target.raise_()
        target.activateWindow()
        return target

    contents_action = help_menu.addAction("Help contents")
    contents_action.setShortcut(QKeySequence(QKeySequence.StandardKey.HelpContents))
    contents_action.triggered.connect(lambda: open_help())
    from .. import howto as _howto

    howto_action = help_menu.addAction(_howto.INDEX_TITLE)  # the task pages' index (pymetrix.howto)
    howto_action.setToolTip("Pages that take one task from start to finish, in the app and in code")
    howto_action.triggered.connect(lambda: open_help(f"pymetrix:topic/{_howto.INDEX}"))
    from .exercises_qt import install as install_exercises

    install_exercises(window, Dock, view, help_menu, settings, app_interface)  # Help → Exercises… and its panel

    def courses() -> None:
        course_dialog(settings, window, current()).exec()
        update_title()  # including course material in the project marks it changed

    help_menu.addAction("Course material…").triggered.connect(courses)
    shortcuts_action = help_menu.addAction("Keyboard shortcuts")
    shortcuts_action.triggered.connect(lambda: shortcuts_dialog(window).show())
    help_menu.addSeparator()  # orientation for users of other packages: where the familiar commands are
    from .. import help as _hp

    for key, label in (("coming-from-eviews", "Coming from EViews"), ("coming-from-iris", "Coming from MATLAB/IRIS")):
        coming = help_menu.addAction(label)
        coming.setToolTip(_hp.topic(key).title)
        coming.triggered.connect(lambda _=False, k=key: open_help(_hp.link("topic", k)))
    help_menu.addSeparator()
    help_menu.addAction("Plugins…").triggered.connect(show_plugins)
    help_menu.addSeparator()
    help_menu.addAction(f"About {NAME}").triggered.connect(lambda: about_dialog(window).exec())
    window.help_menu, window.open_help = help_menu, open_help
    problems = restore_courses(settings)
    if problems:
        window.statusBar().showMessage("Some course folders could not be loaded: " + "; ".join(problems), 10000)

    if settings.contains("state"):
        window.restoreGeometry(settings.value("geometry"))
        window.restoreState(settings.value("state"))
    from .workspaces import install_workspaces

    plugin_docks = install_panels()
    install_workspaces(window, {a.text().replace("&", ""): a.menu() for a in window.menuBar().actions() if a.menu() is not None},
                       {"Objects": objects_dock, "History": history_dock, "Script": script_dock, "Immediate": immediate_dock,
                        "Problems": problems_dock, "Report": report_dock, **plugin_docks},  # by name: Problems' title carries a count
                       view, settings, run_dialog,
                       {"spreadsheet": ("Spreadsheet", lambda: open_sheet()), "save_result": ("Save result", save_result_file),
                        "export_script": ("Export script", export_script)})
    refresh()  # the badges, now the stage bar is there
    window.show_object = show_object  # for scripting and tests
    window.edit_figure = edit_figure
    window.place_note = place_note
    window.open_dropped, window.dropped_files = open_dropped, dropped_files
    window.add_to_report, window.report_view, window.export_report_outline = add_to_report, report_view, export_report_outline
    window.apply_working_set = apply_working_set
    window.table_heatmap = table_heatmap
    window.mdi = mdi
    window.state = state
    window.switch = switch
    window.list_windows = list_windows
    window.view_menu = view
    window.history, window.immediate, window.objects = history, immediate, objects
    window.problems_view, window.status_labels, window.objects_empty = problems_view, status_labels, objects_empty
    window.objects_menu = objects_menu
    window.run_matrix_step, window.open_matrix_grid, window.matrix_menu_for = run_matrix_step, open_matrix_grid, matrix_menu_for
    window.open_editor, window.editors, window.send_editor, window.send_immediate = open_editor, editors, send_editor, send_immediate
    window.recompute, window.refresh, window.save_focused = recompute, refresh, save_focused
    window.calculate_menu, window.compute, window.clean_up_history = calculate, compute, clean_up_history
    window.open_sheet, window.sheets = open_sheet, sheets
    window.open_demo, window.object_filter = open_demo, object_filter
    window.open_project, window.save_project, window.recent_projects = open_project, save, recent_projects

    def show_value(obj: Any, title: str | None = None) -> None:
        """``show()`` in the app: the object's window (its workspace name, if it has one)."""
        name = next((n for n, o in current().objects.items() if o is obj), None)
        if name is not None and title is None:
            refresh()
            show_object(name)
            return
        sub = mdi.addSubWindow(object_view(obj, name=name or ""))
        sub.setAttribute(QtNS.WidgetAttribute.WA_DeleteOnClose)
        sub.setWindowTitle(title or f"{kind_of(obj)} (shown)")
        sub.setWindowIcon(object_icon(obj))
        sub.resize(680, 460)
        sub.show()
        mdi.setActiveSubWindow(sub)

    show_token = object()  # this window in _SHOWING
    _show_in(show_token, show_value)
    window.destroyed.connect(lambda *_: _show_closed(show_token))
    window.show_value = show_value
    # autosave: a copy of a session with unsaved changes every few minutes, offered back after a crash (pymetrix.recovery)
    import time as _time
    import uuid

    from PySide6.QtCore import QTimer

    from .. import recovery

    autosave = {"key": uuid.uuid4().hex, "token": None, "last": _time.time()}

    def autosave_minutes() -> float:
        try:
            return float(settings.value("autosave_minutes", 2))
        except (TypeError, ValueError):
            return 2.0

    def autosave_now(force: bool = False) -> bool:
        """Write the copy if there are unsaved changes since the last one, and the interval has passed (or ``force``)."""
        recovery.heartbeat(autosave["key"])  # this app is running: its copy is not an orphan
        s = current()
        if not s.dirty or busy() or (autosave_minutes() <= 0 and not force):
            return False
        token = (len(s.history), tuple(sorted(s._snapshot().items())), str(s.path))
        if token == autosave["token"] and not force:
            return False
        if not force and _time.time() - autosave["last"] < autosave_minutes() * 60:
            return False
        try:
            recovery.write(s, autosave["key"], title=Path(s.path).name if s.path else "")
        except Exception as exc:  # noqa: BLE001 - autosave must never interrupt the work
            window.statusBar().showMessage(f"Autosave failed: {exc}", 10000)
            return False
        autosave.update(token=token, last=_time.time())
        return True

    def forget_autosave() -> None:
        recovery.discard(autosave["key"])
        autosave.update(token=None, last=_time.time())

    def stop_autosave() -> None:
        timer = getattr(window, "autosave_timer", None)
        if timer is not None:
            timer.stop()
        forget_autosave()

    def offer_recovery(answer: Any = None) -> int:
        """Offer back the copies left by apps that stopped without closing; ``answer(recovery)`` -> "recover" |
        "discard" | "later" replaces the question (tests). Returns how many were recovered."""
        from ..commands import load_project

        recovered = 0
        for rec in recovery.orphans(exclude=autosave["key"]):
            if answer is not None:
                choice = answer(rec)
            else:
                box = Qt.QMessageBox(window)
                box.setWindowTitle("Recover unsaved work")
                box.setText(f"{NAME} stopped without closing. A copy of {rec.describe()} was kept.")
                box.setInformativeText("Recover it now? It opens with your unsaved changes; save it to keep them.")
                yes = box.addButton("Recover", Qt.QMessageBox.ButtonRole.AcceptRole)
                no = box.addButton("Discard", Qt.QMessageBox.ButtonRole.DestructiveRole)
                box.addButton("Later", Qt.QMessageBox.ButtonRole.RejectRole)
                box.exec()
                choice = "recover" if box.clickedButton() is yes else "discard" if box.clickedButton() is no else "later"
            if choice == "recover" and not recovered and confirm_discard():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        restored = load_project(rec.path)
                except Exception as exc:  # noqa: BLE001 - say why, keep the copy for another try
                    Qt.QMessageBox.critical(window, *titled("Could not recover", str(exc)))
                    continue
                restored.path = Path(rec.original) if rec.original else None
                restored.dirty = True  # recovered changes are unsaved until saved
                switch(restored)
                recovery.discard(rec.key)
                update_title()
                window.statusBar().showMessage(f"Recovered {rec.describe()}: save it to keep the changes", 15000)
                recovered += 1
            elif choice == "discard":
                recovery.discard(rec.key)
        return recovered

    from ..plugins import load_problems

    failed = load_problems()
    if failed:
        window.statusBar().showMessage(f"{len(failed)} plugin{'s' if len(failed) != 1 else ''} could not load: "
                                       f"{', '.join(failed)} (Help → Plugins; Problems lists why)", 20000)
    autosave_timer = QTimer(window)
    autosave_timer.setInterval(60_000)
    autosave_timer.timeout.connect(lambda: autosave_now())
    autosave_timer.start()
    recovery.heartbeat(autosave["key"])
    window.autosave_timer, window.autosave_now, window.offer_recovery = autosave_timer, autosave_now, offer_recovery
    window.autosave_key = autosave["key"]
    from PySide6.QtGui import QAction

    autosave_toggle = QAction("Autosave for recovery", file_menu)
    file_menu.insertAction(quit_separator, autosave_toggle)  # before Exit, which stays last
    autosave_toggle.setCheckable(True)
    autosave_toggle.setChecked(autosave_minutes() > 0)
    autosave_toggle.setToolTip("every 2 minutes, a copy of unsaved work in ~/.pymetrix/recovery, offered back after a crash")
    autosave_toggle.toggled.connect(lambda on: settings.setValue("autosave_minutes", 2 if on else 0))
    window.autosave_toggle = autosave_toggle

    for top in window.menuBar().actions():  # every line of every menu gets an access key (Alt+M, then K: Kalman filter)
        if top.menu() is not None:
            keep_access_keys(top.menu())
    window.start_page = start_page(window, run_dialog, open_project, open_help, recent_projects(), open_demo) if not current().objects else None
    refresh()
    return window


# keys that belong to a window or a field rather than a menu: listed with the menus' shortcuts
OTHER_SHORTCUTS = (
    ("Command dialogs", "Help on the command", "F1"),
    ("Command dialogs", "Run the step", "Enter"),
    ("Command dialogs", "Close (or, while the step runs, cancel it)", "Esc"),
    ("Spreadsheet edit line", "Accept the value and move down", "Enter"),
    ("Spreadsheet edit line", "Put the cell's value back", "Esc"),
    ("Code windows", "Find text", "Ctrl+F"),
    ("Immediate", "Earlier and later lines typed", "Up / Down"),
    ("Figures", "Take away the box a click shows; stop a drag or Draw an arrow / Mark a change; clear the selection", "Esc"),
    ("Figures", "Remove the selected note, arrow, text, line, band or change marker", "Delete"),
    ("Figures", "A menu for what is under the mouse: format, words, order, delete; format series or axes", "Right-click (Ctrl+click on a Mac)"),
    ("Figures", "Move only across, or only up and down, while dragging", "Shift"),
)


def shortcut_list(window: Any) -> list[tuple[str, str, str]]:
    """Every keyboard shortcut of the main window: (where, what, keys), from its menus as they are (so the list never
    goes out of date), then the keys of dialogs and panels, all in the platform's own words (⌘F on a Mac)."""
    found: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()

    def walk(menu: Any, where: str) -> None:
        for action in menu.actions():
            if action.menu() is not None:
                walk(action.menu(), f"{where} → {action.menu().title().replace('&', '')}")
            elif not action.isSeparator():
                keys = ", ".join(key_text(k) for k in action.shortcuts() if not k.isEmpty())
                label = action.text().replace("&", "").rstrip(".…")
                if keys and (label, keys) not in seen:
                    seen.add((label, keys))
                    found.append((where, label, keys))

    for top in window.menuBar().actions():
        if top.menu() is not None:
            walk(top.menu(), top.menu().title().replace("&", ""))
    # the other keys in the same words as the menus' (Ctrl+F is ⌘F on a Mac, where Qt's Ctrl is the Command key)
    return found + [(where, what, key_text(keys) if keys.startswith("Ctrl+") else keys)
                    for where, what, keys in OTHER_SHORTCUTS]


def shortcuts_dialog(window: Any) -> Any:
    """Help → Keyboard shortcuts: a filterable table of every shortcut."""
    Qt = _qt()
    dialog = Qt.QDialog(window)
    dialog.setWindowTitle("Keyboard shortcuts")
    layout = Qt.QVBoxLayout(dialog)
    metrics.set_margins(layout, "dialog_margin")
    search = Qt.QLineEdit()
    search.setPlaceholderText("Filter by command, menu or key")
    search.setClearButtonEnabled(True)
    layout.addWidget(search)
    rows = shortcut_list(window)
    table = Qt.QTableWidget(len(rows), 3)
    table.setHorizontalHeaderLabels(["Keys", "Command", "Where"])
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(Qt.QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(Qt.QAbstractItemView.SelectionBehavior.SelectRows)
    for r, (where, label, keys) in enumerate(rows):
        for c, text in enumerate((keys, label, where)):
            table.setItem(r, c, Qt.QTableWidgetItem(text))
    table.resizeColumnsToContents()
    header = table.horizontalHeader()  # Keys and Where as wide as they need, Command the rest
    header.setStretchLastSection(False)
    header.setSectionResizeMode(0, Qt.QHeaderView.ResizeMode.ResizeToContents)
    header.setSectionResizeMode(1, Qt.QHeaderView.ResizeMode.Stretch)
    header.setSectionResizeMode(2, Qt.QHeaderView.ResizeMode.ResizeToContents)
    layout.addWidget(table)

    def narrow(text: str) -> None:
        words = text.lower().split()
        for r, row in enumerate(rows):
            table.setRowHidden(r, not all(any(w in part.lower() for part in row) for w in words))

    search.textChanged.connect(narrow)
    buttons = Qt.QDialogButtonBox(Qt.QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    from .fit import fit_to_screen

    fit_to_screen(dialog, 620, 520)
    dialog.search, dialog.table = search, table
    return dialog


DATA_FILES = (".csv", ".xlsx", ".xls", ".parquet", ".dta", ".sav", ".json", ".txt")  # dropped on the window: Load data
START_SIZE = (620, 640)  # the Start window's natural size, at most: where the workspace has room for it
START_LEAST = (360, 260)  # and its least, unless the workspace is smaller still (its content then scrolls)
FOLDED, UNFOLDED = "\u25b8", "\u25be"  # ▸ and ▾ before a folded and an unfolded section's toggle


def chart_gallery_file(module: str | Path = __file__) -> Path | None:
    """The chart gallery notebook (every kind of figure with its code), when the app runs from a checkout of the
    repository (``module``, this file, in its ``src/pymetrix/ui``); the package (wheel) holds no examples, so an
    installed copy has none: None."""
    found = Path(module).resolve().parents[3] / "examples" / "gallery" / "chart_gallery.ipynb"
    return found if found.is_file() else None


def open_file(path: Any) -> bool:
    """Open a file in the system's default application for it (a notebook in Jupyter or an editor)."""
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices

    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def _spacing_across(layout: Any, widget: Any) -> int:
    """The pixels a layout leaves between two push buttons side by side: its own spacing, or its style's."""
    from PySide6.QtCore import Qt as QtNS
    from PySide6.QtWidgets import QSizePolicy

    spacing = layout.horizontalSpacing() if hasattr(layout, "horizontalSpacing") else layout.spacing()
    if spacing < 0:
        button = QSizePolicy.ControlType.PushButton
        spacing = widget.style().layoutSpacing(button, button, QtNS.Orientation.Horizontal)
    return max(spacing, 0)


BREAK = "\u200b"  # a zero-width space: where a line may break in a word with no space of its own


def breakable(text: str) -> str:
    """``text`` with a place to break a line after each underscore and hyphen: a list entry wraps at its width even
    where a name (``policy_and_nonlinear``) or a span (``2025Q1-2026Q4``, no break before a digit) is wider than the
    list, which a wrap at spaces alone cannot do."""
    return text.replace("_", "_" + BREAK).replace("-", "-" + BREAK)


def start_page(window: Any, run_dialog: Any, open_project: Any, open_help: Any, recent: list[str] | None = None,
               open_demo: Any = None) -> Any:
    """A first window for an empty workspace: where to begin (data, a recent project, a demo), the workspaces, and
    that every step becomes Python."""
    Qt = _qt()
    from PySide6.QtCore import Qt as QtNS

    from ..commands import LoadData, LoadExample

    from .fit import EntryList, KeepInViewport, WrapButton, WrapLabel, scroll_area

    page = Qt.QWidget()
    layout = Qt.QVBoxLayout(page)
    metrics.set_margins(layout, "normal")  # a page in the workspace: Qt's 9 for a page, on the grid (a dialog's 12 cost a
    # narrow workspace the room that kept the page from scrolling sideways)
    text = WrapLabel(  # short: on a small or scaled screen the demos and the ways to start come first
        f"<h2>{NAME}</h2>"
        "<p>Open a demo project to see finished work, step by step, or start with data: the example economy used in "
        "the course labs, a file of your own, or a saved project.</p>")
    text.setWordWrap(True)
    text.setTextFormat(QtNS.TextFormat.RichText)
    layout.addWidget(text)
    # a short window scrolls rather than squashes its lists; a narrow one never scrolls sideways: everything on the
    # page wraps to its width, its text breaking a word where the word alone is wider (WrapLabel: a workspace
    # narrower than the page's longest word on Windows CI, all panels open)
    sub = window.mdi.addSubWindow(scroll_area(page, sideways=False))
    sub.setAttribute(QtNS.WidgetAttribute.WA_DeleteOnClose)
    sub.setWindowTitle("Start")
    buttons = {}
    row = Qt.QWidget()  # the ways to start, two by two (half the height of a column), one under another when narrow,
    # their text wrapped where even one column is narrower than it (a large font in a narrow workspace): the window
    # never held wider than the workspace by a button
    grid = Qt.QGridLayout(row)
    grid.setContentsMargins(0, 0, 0, 0)
    for label, action in (("Load example data…", lambda: run_dialog(LoadExample)), ("Load a data file…", lambda: run_dialog(LoadData)),
                          ("Open a project…", open_project), ("Help contents", lambda: open_help())):
        button = WrapButton(label)
        button.clicked.connect(lambda _=False, a=action: (sub.close(), a()))
        buttons[label] = button
    layout.addWidget(row)

    def arrange(width: int) -> None:
        # two columns where they fit beside each other in ``width`` (the scroll area's viewport): each column as wide as
        # its widest button with its text on one line (its size hint, not its least size: that wraps the text), the
        # grid's spacing across between them and the page's margins around. Not spacing(), which is -1 where a
        # style's spacing across and down differ or it spaces by the kind of widget
        least = [b.sizeHint().expandedTo(b.minimumSize()).width() for b in buttons.values()]
        margins = layout.contentsMargins()
        both = (max(least[0::2]) + max(least[1::2]) + _spacing_across(grid, row)
                + margins.left() + margins.right())
        columns = 2 if width >= both else 1
        if getattr(row, "columns", None) == columns:
            return
        row.columns = columns
        for k, button in enumerate(buttons.values()):
            grid.removeWidget(button)
            grid.addWidget(button, k // columns, k % columns)

    from PySide6.QtCore import QObject

    class _FollowWidth(QObject):
        def eventFilter(self, watched: Any, event: Any) -> bool:  # noqa: N802 - Qt's name
            if event.type() == event.Type.Resize:
                arrange(watched.width())
            return False

    arrange(START_SIZE[0])
    gallery_file = chart_gallery_file()
    gallery = None
    if gallery_file is not None:  # from a checkout: a link that opens the notebook (an installed copy has none)
        gallery = WrapLabel('<a href="gallery">Chart gallery</a>: every kind of figure pymetrix draws, with its code '
                            "(a notebook in the repository's examples)")
        gallery.setTextFormat(QtNS.TextFormat.RichText)
        gallery.setWordWrap(True)
        gallery.setToolTip(str(gallery_file))
        gallery.setTextInteractionFlags(QtNS.TextInteractionFlag.LinksAccessibleByMouse
                                        | QtNS.TextInteractionFlag.LinksAccessibleByKeyboard)
        gallery.linkActivated.connect(lambda _=None, f=gallery_file: open_file(f))
        layout.addWidget(gallery)  # under the buttons, so it shows whatever the window's height
    lists: dict[str, Any] = {}

    def listing(title: str, items: list[tuple[str, str, Any]]) -> None:
        """A heading and a list: a click (or Enter) opens an item. Long entries wrap to the list's width in any style
        (the full text is also in the tooltip): no sideways scrolling."""
        if not items:
            return
        heading = WrapLabel(f"<b>{title}</b>")
        heading.setWordWrap(True)  # nothing on the page holds it wider than a word
        metrics.space_above(heading)  # the same room above every section's heading
        layout.addWidget(heading)
        widget = EntryList([(breakable(text), tip) for text, tip, _ in items])
        widget.setAccessibleName(title)
        widget.activated.connect(lambda row: (sub.close(), items[row][2]()))
        layout.addWidget(widget)
        lists[title] = widget

    if open_demo is not None:  # first: a demo shows the app at work at once
        from ..demos import demos

        listing("Demo projects", [(f"{n}: {d}", f"{n}: {d}", lambda n=n: open_demo(n)) for n, d in demos().items()])
    from pathlib import Path as _Path

    listing("Recent projects", [(_Path(p).name, p, lambda p=p: open_project(p)) for p in (recent or [])[:6]])
    about = WrapButton(f"{FOLDED} About this window")  # the tour of the window, folded: there when wanted, not
    about.setToolTip("the workspaces, Find anything, and the Python every step records")  # pushing the demos down
    about.setCheckable(True)
    about.setFlat(True)
    tour = WrapLabel(
        "<p>The bar above picks a <b>workspace</b> for each stage: <b>Explore</b> the data, <b>Estimate</b>, "
        "<b>Model</b>, <b>Forecast</b>, <b>Report</b>, or <b>All</b>; <b>Next ▸</b> on a result leads on. "
        f"<b>{key_text('Ctrl+K')}</b> finds any command. Every step is recorded as Python in the <b>History</b> and "
        "<b>Script</b> windows, and <b>File → Export script</b> saves it. Files can be dropped on the window to open them.</p>")
    tour.setWordWrap(True)
    tour.setTextFormat(QtNS.TextFormat.RichText)
    tour.setVisible(False)

    def unfold(on: bool) -> None:
        tour.setVisible(on)
        about.setText(f"{UNFOLDED if on else FOLDED} About this window")

    about.toggled.connect(unfold)
    layout.addWidget(about, 0, QtNS.AlignmentFlag.AlignLeft)
    layout.addWidget(tour)
    layout.addStretch()

    def natural() -> Any:
        """Its natural size: START_SIZE's width (or the workspace's, if narrower), and the height its content takes
        at that width (at most START_SIZE's height), with the window's frame and title bar."""
        from PySide6.QtCore import QSize

        from .fit import VIEW_MARGIN

        margins = sub.contentsMargins()
        chrome_w, chrome_h = margins.left() + margins.right(), margins.top() + margins.bottom()
        room = sub.mdiArea().viewport().width() - 2 * VIEW_MARGIN
        width = max(min(START_SIZE[0], room), START_LEAST[0])
        inner = layout.totalHeightForWidth(width - chrome_w) if layout.hasHeightForWidth() else page.sizeHint().height()
        return QSize(width, min(max(inner + chrome_h, START_LEAST[1]), START_SIZE[1]))

    from PySide6.QtCore import QSize

    sub.resize(*START_SIZE)
    sub.show()
    sub.keep_fitted = KeepInViewport(sub, natural, QSize(*START_LEAST))  # sized and centred in the workspace's free area
    follow = _FollowWidth(sub)
    sub.widget().viewport().installEventFilter(follow)  # two columns of buttons, or one, as the window's width allows
    arrange(sub.widget().viewport().width())
    sub.buttons, sub.lists, sub.gallery, sub.about, sub.tour, sub.follow_width = buttons, lists, gallery, about, tour, follow
    return sub


def _arguments(argv: list[str]) -> Any:
    import argparse

    parser = argparse.ArgumentParser(prog="pymetrix-gui", description="The PyMetrix desktop app.")
    parser.add_argument("project", nargs="?", help="a project file (.pymx) to open")
    parser.add_argument("--demo", help="open a demo project: " + ", ".join(__import__("pymetrix.demos", fromlist=["demos"]).demos()))
    parser.add_argument("--maximized", action="store_true", help="fill the screen (e.g. in a browser desktop)")
    parser.add_argument("--shortcut", action="store_true", help="make a desktop (and Start menu) shortcut to the app, and stop")
    parser.add_argument("--latest", action="store_true",
                        help="update a git checkout (git pull --ff-only) before starting; with --shortcut, make a "
                             "shortcut that does so each time")
    ours, skip = [], False
    for arg in argv:  # Qt's own options (-platform offscreen, -style fusion, ...) are left for Qt
        if skip:
            skip = False
        elif arg.startswith("-") and not arg.startswith("--") and len(arg) > 2:
            skip = arg.lstrip("-") in _QT_VALUE_OPTIONS
        else:
            ours.append(arg)
    args, _ = parser.parse_known_args(ours)
    return args


_QT_VALUE_OPTIONS = {"platform", "platformpluginpath", "platformtheme", "plugin", "style", "stylesheet", "session", "display",
                     "geometry", "title", "qwindowgeometry", "qwindowicon", "qwindowtitle", "qmljsdebugger"}


def about_dialog(parent: Any) -> Any:
    """Help → About: the name, version and copyright the opening screen shows."""
    from pymetrix_app import about_lines

    Qt = _qt()
    name, number, tagline, rights = about_lines()
    box = Qt.QMessageBox(parent)
    box.setWindowTitle(f"About {name}")
    box.setText(f"<h2>{name}</h2><p>{number}</p><p>{tagline}</p><p>{rights}</p>")
    return box


def main(argv: list[str] | None = None, app: Any = None, splash: Any = None) -> None:
    """The desktop app. ``pymetrix-gui`` (``pymetrix_app.main``) passes the QApplication and the opening screen it
    showed before pymetrix loaded; started otherwise (``python -m pymetrix.ui.qt``), this makes them."""
    args = _arguments(sys.argv[1:] if argv is None else argv)
    if args.shortcut:  # before Qt: making a shortcut opens no window
        from .shortcut import main as make_shortcut

        make_shortcut(latest=args.latest)
        return
    Qt = _qt()
    app = app or Qt.QApplication.instance() or Qt.QApplication(sys.argv)
    from . import collector

    collector.install(app)  # Python's cyclic collector on the GUI thread only, from the start
    from pymetrix_app import identify

    from ..commands import events

    events.INTERFACE = "app"  # steps taken in the app are recorded as the person's, through the app

    identify(app)
    if splash is None:
        from pymetrix_app import splash as opening_screen

        splash = opening_screen(app)

    def loading(what: str) -> None:
        from PySide6.QtCore import Qt as QtNS
        from PySide6.QtGui import QColor

        splash.showMessage(what, QtNS.AlignmentFlag.AlignBottom | QtNS.AlignmentFlag.AlignLeft, QColor(theme.SPLASH_TEXT))
        app.processEvents()

    if args.demo:
        loading(f"Opening the {args.demo} demo…")
        from ..demos import demo

        session = demo(args.demo)
    elif args.project:
        loading(f"Opening {Path(args.project).name}…")
        from ..commands import load_project

        session = load_project(args.project)
    else:
        session = Session()
    loading("Building the window…")
    window = main_window(session)
    from .fit import fit_to_screen

    fit_to_screen(window, 1100, 700)  # a laptop at 1024 by 600 gets a window that fits it
    window.showMaximized() if args.maximized else window.show()
    splash.finish(window)
    from PySide6.QtCore import QTimer

    QTimer.singleShot(0, window.offer_recovery)  # work left by a crash, once the window is up
    sys.exit(_run(app))


def _run(app: Any) -> int:
    """Run the app until it quits, then stop Java if TRAMO-SEATS started it, while Qt is still whole: on Windows, Java
    still running as Qt's windows were destroyed crashed the process on the way out."""
    code = app.exec()
    jdemetra = sys.modules.get("pymetrix_seasonal.jdemetra")  # the seasonal plugin's Java, if it was started
    if jdemetra is not None:
        jdemetra.shutdown()
    return code


if __name__ == "__main__":
    main()
