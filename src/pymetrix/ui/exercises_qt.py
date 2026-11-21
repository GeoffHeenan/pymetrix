"""Exercise packs in the desktop app: Help → Exercises… and the Exercise panel (``pymetrix.exercises``).

- ``exercises_dialog`` lists the packs: those installed plugins ship, and folders added with **Open folder…**
  (remembered in the app's settings, as course folders are). Choosing an exercise and **Start** opens it in the panel.
- ``ExercisePane`` docks on the right: the exercise's introduction, then each step with its instructions, **Show hint**
  (one hint more at each press), **Show the code** (the step as code, folded, with Copy, as under a dialog's fields)
  and **Check**, which runs the step's checks against the open session and shows each, green or red, with its message.
- When an exercise starts, its datasets load into the session (as File → Load data… or Load example data… steps,
  recorded and undoable) once the user agrees.

Nothing in a pack runs here: checks are data, and ``solution.py`` is left to ``pymetrix-exercises check``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import exercises as ex

__all__ = ["FOLDERS_KEY", "ExercisePane", "available_packs", "exercises_dialog", "fail_color", "pass_color", "result_text"]

FOLDERS_KEY = "exercise_folders"  # the folders added with Open folder…, in the app's settings


def pass_color() -> str:
    """Green for a check that passes: the theme's "success", readable on the current palette."""
    from . import theme

    return theme.name("success")


def fail_color() -> str:
    """Red for a check that fails: the theme's "error"."""
    from . import theme

    return theme.name("error")


def _qt() -> Any:
    from PySide6 import QtWidgets

    return QtWidgets


def _folders(settings: Any) -> list[str]:
    if settings is None:
        return []
    return [str(f) for f in (settings.value(FOLDERS_KEY, [], type=list) or [])]


def available_packs(settings: Any = None) -> tuple[list[ex.Pack], list[str]]:
    """The installed packs, then the folders added in the app; and the problems met reading them."""
    packs = ex.installed_packs()
    problems = [f"{label}: {why}" for label, why in ex.problems().items()]
    for folder in _folders(settings):
        try:
            packs.append(ex.load_pack(folder))
        except Exception as exc:  # noqa: BLE001 - a folder moved or broken: said, not raised
            problems.append(f"{folder}: {exc}")
    return packs, problems


def result_text(r: ex.CheckResult) -> str:
    """One check's line as the panel shows it (rich text)."""
    import html

    mark = "✓" if r.passed else "✗"
    from . import theme

    color = pass_color() if r.passed else fail_color()
    text = f'<span style="color: {color}"><b>{mark} {html.escape(r.id)}</b></span>'
    if not r.passed and r.message:
        text += f" — {html.escape(r.message)}"
    if r.detail:
        text += f'<br><span style="{theme.muted_css()}">{html.escape(r.detail)}</span>'
    return text


def _make_pane_class() -> type:
    import html

    from PySide6 import QtCore, QtWidgets

    from . import metrics
    from .text_size import fixed_font

    Qt = QtCore.Qt

    class StepBox(QtWidgets.QFrame):
        """One step: its title and instructions, Show hint, Show the code, Check, and the check's results."""

        def __init__(self, pane: Any, step: ex.Step) -> None:
            super().__init__(objectName=f"exercise-step-{step.number}")
            self.pane, self.step, self.hints_shown = pane, step, 0
            self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
            layout = QtWidgets.QVBoxLayout(self)
            metrics.set_margins(layout, "normal")
            title = QtWidgets.QLabel(f"<b>Step {step.number}: {step.title}</b>", objectName="exercise-step-title")
            title.setWordWrap(True)
            layout.addWidget(title)
            self.text = QtWidgets.QLabel(objectName="exercise-step-text")
            self.text.setTextFormat(Qt.TextFormat.MarkdownText)
            self.text.setWordWrap(True)
            self.text.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
            self.text.setText(step.text)
            layout.addWidget(self.text)
            self.hint_label = QtWidgets.QLabel(objectName="exercise-hints")
            self.hint_label.setTextFormat(Qt.TextFormat.MarkdownText)
            self.hint_label.setWordWrap(True)
            self.hint_label.setVisible(False)
            layout.addWidget(self.hint_label)
            self.code_row = QtWidgets.QWidget(objectName="exercise-code")
            code_line = QtWidgets.QHBoxLayout(self.code_row)
            code_line.setContentsMargins(0, 0, 0, 0)
            self.code = QtWidgets.QPlainTextEdit(objectName="exercise-code-text")
            self.code.setReadOnly(True)
            self.code.setFont(fixed_font())
            self.code.setPlainText(step.code)
            self.code.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
            lines = max(1, step.code.count("\n") + 1)
            self.code.setFixedHeight(self.code.fontMetrics().lineSpacing() * min(lines, 8) + 2 * self.code.frameWidth()
                                     + int(self.code.document().documentMargin() * 2) + 4)
            copy = QtWidgets.QPushButton("Copy", objectName="exercise-copy")
            copy.setToolTip("copy the code")
            copy.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(step.code))
            code_line.addWidget(self.code, 1)
            code_line.addWidget(copy, 0, Qt.AlignmentFlag.AlignTop)
            self.code_row.setVisible(False)
            row = QtWidgets.QHBoxLayout()
            self.hint_button = QtWidgets.QPushButton("Show hint", objectName="exercise-hint")
            self.hint_button.setEnabled(bool(step.hints))
            self.hint_button.setToolTip("a hint for this step" if step.hints else "this step has no hints")
            self.hint_button.clicked.connect(self.show_hint)
            self.code_toggle = QtWidgets.QToolButton(objectName="exercise-code-toggle")
            self.code_toggle.setText("Show the code")
            self.code_toggle.setCheckable(True)
            self.code_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            self.code_toggle.setArrowType(Qt.ArrowType.RightArrow)
            self.code_toggle.setAutoRaise(True)
            self.code_toggle.setEnabled(bool(step.code))
            self.code_toggle.setToolTip("the same step in Python, as the Immediate window runs it" if step.code
                                        else "this step has no code")
            self.code_toggle.toggled.connect(self.show_code)
            self.check_button = QtWidgets.QPushButton("Check", objectName="exercise-check")
            checks = pane.exercise.checks_for(step.number)
            self.check_button.setEnabled(bool(checks))
            self.check_button.setToolTip(f"check this step against the session ({len(checks)} check{'s' * (len(checks) != 1)})"
                                         if checks else "this step has no checks")
            self.check_button.clicked.connect(self.run_checks)
            row.addWidget(self.hint_button)
            row.addWidget(self.code_toggle)
            row.addStretch(1)
            row.addWidget(self.check_button)
            layout.addLayout(row)
            layout.addWidget(self.code_row)
            self.results = QtWidgets.QLabel(objectName="exercise-results")
            self.results.setTextFormat(Qt.TextFormat.RichText)
            self.results.setWordWrap(True)
            self.results.setVisible(False)
            layout.addWidget(self.results)
            self.last: list[ex.CheckResult] = []

        def show_hint(self) -> None:
            self.hints_shown = min(self.hints_shown + 1, len(self.step.hints))
            shown = self.step.hints[:self.hints_shown]
            self.hint_label.setText("\n\n".join(f"*Hint {i}.* {h}" if len(self.step.hints) > 1 else f"*Hint.* {h}"
                                                for i, h in enumerate(shown, 1)))
            self.hint_label.setVisible(bool(shown))
            more = self.hints_shown < len(self.step.hints)
            self.hint_button.setText("Another hint" if more else "Show hint")
            self.hint_button.setEnabled(more)

        def show_code(self, on: bool) -> None:
            self.code_toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
            self.code_toggle.setText("Hide the code" if on else "Show the code")
            self.code_row.setVisible(on)

        def run_checks(self) -> list[ex.CheckResult]:
            try:
                self.last = ex.check(self.pane.exercise, self.pane.session(), self.step.number)
            except Exception as exc:  # noqa: BLE001 - shown, never raised into the event loop
                self.last = []
                self.results.setText(f'<span style="color: {fail_color()}">The checks could not run: {exc}</span>')
                self.results.setVisible(True)
                return self.last
            n = sum(r.passed for r in self.last)
            head = (f'<span style="color: {pass_color()}"><b>All {n} checks pass.</b></span>' if n == len(self.last)
                    else f"<b>{n} of {len(self.last)} checks pass.</b>")
            self.results.setText(head + "<br>" + "<br>".join(result_text(r) for r in self.last))
            self.results.setVisible(True)
            self.pane.checked(self.step.number, self.last)
            return self.last

    class ExercisePane(QtWidgets.QWidget):
        """The Exercise panel: one exercise, every step in view."""

        def __init__(self, app: Any, confirm: Callable[[str], bool] | None = None) -> None:
            super().__init__(objectName="exercise-pane")
            self.app = app
            self.exercise: ex.Exercise | None = None
            self.boxes: list[StepBox] = []
            self.confirm = confirm or self._ask
            outer = QtWidgets.QVBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            self.scroll = QtWidgets.QScrollArea(objectName="exercise-scroll", widgetResizable=True)
            self.scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            outer.addWidget(self.scroll, 1)
            self.empty = QtWidgets.QLabel("No exercise open: choose one with Help → Exercises….", objectName="exercise-empty")
            self.empty.setWordWrap(True)
            self._body()

        def _body(self) -> None:
            body = QtWidgets.QWidget(objectName="exercise-body")
            self.layout_ = QtWidgets.QVBoxLayout(body)
            metrics.set_margins(self.layout_, "normal")
            self.scroll.setWidget(body)  # the old body is deleted with it

        def session(self) -> Any:
            return self.app.session

        @staticmethod
        def _ask(question: str) -> bool:
            box = QtWidgets.QMessageBox
            return box.question(None, "Start the exercise", question, box.StandardButton.Yes | box.StandardButton.No,
                                box.StandardButton.Yes) == box.StandardButton.Yes

        def start(self, exercise: ex.Exercise, load: bool = True) -> None:
            """Show ``exercise``; with ``load``, offer to load its data into the session first."""
            self.exercise = exercise
            self._body()
            pack = exercise.pack
            head = QtWidgets.QLabel(objectName="exercise-title")
            head.setWordWrap(True)
            about = " · ".join(x for x in (exercise.level, f"about {exercise.minutes} minutes" if exercise.minutes else "") if x)
            head.setText(f"<b>{html.escape(exercise.title)}</b>" + (f"<br><i>{html.escape(pack.title)}</i>" if pack else "")
                         + (f" · {html.escape(about)}" if about else "")
                         + (f"<br>Classification: {html.escape(pack.classification)}" if pack and pack.classification else ""))
            self.layout_.addWidget(head)
            if pack is not None and pack.requirement_problem:
                warn = QtWidgets.QLabel(pack.requirement_problem, objectName="exercise-requirement")
                warn.setWordWrap(True)
                warn.setStyleSheet(f"color: {fail_color()}")
                self.layout_.addWidget(warn)
            self.data_label = QtWidgets.QLabel(objectName="exercise-data")
            self.data_label.setWordWrap(True)
            self.layout_.addWidget(self.data_label)
            self.model_buttons = {}
            for name in exercise.models:  # each opens its import dialog filled in with the pack's file and its name
                row = QtWidgets.QHBoxLayout()
                shown = pack.model_name(name)  # the session name: [models.<stem>] name in pack.toml, else the stem
                label = QtWidgets.QLabel(f"Model: {shown} ({pack.models[name].name})", objectName="exercise-model")
                button = QtWidgets.QPushButton("Import…", objectName=f"exercise-import-{name}")
                button.setToolTip(f"open the import dialog with the pack's file {pack.models[name].name}: run it to add {shown}")
                button.clicked.connect(lambda _=False, n=name: self.import_model(n))
                row.addWidget(label, 1)
                row.addWidget(button)
                self.layout_.addLayout(row)
                self.model_buttons[name] = button
            if exercise.intro:
                intro = QtWidgets.QLabel(objectName="exercise-intro")
                intro.setTextFormat(Qt.TextFormat.MarkdownText)
                intro.setWordWrap(True)
                intro.setText(exercise.intro)
                self.layout_.addWidget(intro)
            self.boxes = [StepBox(self, s) for s in exercise.steps]
            for box in self.boxes:
                self.layout_.addWidget(box)
            if exercise.closing:
                closing = QtWidgets.QLabel(objectName="exercise-closing")
                closing.setTextFormat(Qt.TextFormat.MarkdownText)
                closing.setWordWrap(True)
                closing.setText(exercise.closing)
                self.layout_.addWidget(closing)
            self.layout_.addStretch(1)
            self.data_label.setText(self._data_text(loaded=False))
            if load and exercise.data:
                self.load_data()

        def _data_text(self, loaded: bool) -> str:
            names = self.exercise.data if self.exercise else ()
            if not names:
                return ""
            listed = ", ".join(names)
            return f"Data: {listed}" + (" (loaded)" if loaded else " (not loaded: load it, or open a project that has it)")

        def load_data(self) -> bool:
            """Ask, then load the exercise's datasets as recorded steps; True if they loaded."""
            exercise = self.exercise
            if exercise is None or not exercise.data:
                return False
            have = [n for n in exercise.data if n in getattr(self.session(), "objects", {})]
            question = (f"Load the data for “{exercise.title}” into the session: {', '.join(exercise.data)}?"
                        + (f"\n\nThis replaces the session's {', '.join(have)}." if have else "")
                        + "\n\nEach dataset loads as a step, in History and the script; Undo takes it back.")
            if not self.confirm(question):
                self.app.status("Exercise: the data were not loaded")
                return False
            try:
                for cmd in ex.data_commands(exercise):
                    self.app.run(cmd)
            except Exception as exc:  # noqa: BLE001 - said in the panel and the status bar
                self.data_label.setText(f"The data could not be loaded: {exc}")
                self.app.status(f"Exercise: the data could not be loaded: {exc}")
                return False
            self.data_label.setText(self._data_text(loaded=True))
            return True

        def import_model(self, name: str) -> None:
            """Open the dialog that imports the pack's model ``name`` (Model → Import Dynare file…, Import IRIS
            file…, or the Model editor), filled in with its file; the user runs it as any step."""
            try:
                cmd = ex.model_command(self.exercise.pack, name)
                self.app.open_dialog(type(cmd), cmd)
            except Exception as exc:  # noqa: BLE001 - said in the status bar
                self.app.status(f"Exercise: the model {name} could not be opened: {exc}")

        def checked(self, step: int, results: list[ex.CheckResult]) -> None:
            passed = sum(r.passed for r in results)
            self.app.status(f"Exercise, step {step}: {passed} of {len(results)} checks pass")

    return ExercisePane


_PANE: type | None = None


def __getattr__(name: str) -> Any:
    if name == "ExercisePane":
        global _PANE
        if _PANE is None:
            _PANE = _make_pane_class()
        return _PANE
    raise AttributeError(name)


def create_pane(app: Any, confirm: Callable[[str], bool] | None = None) -> Any:
    """The Exercise panel (a QWidget) over the app's ``AppInterface``; ``confirm(question)`` asks before data load."""
    return __getattr__("ExercisePane")(app, confirm)


def exercises_dialog(parent: Any = None, settings: Any = None, start: Callable[[ex.Exercise], None] | None = None) -> Any:
    """Help → Exercises…: the packs and their exercises, Open folder… to add a pack, Start to open the chosen one."""
    Qt = _qt()
    from PySide6.QtCore import Qt as QtNS

    from .fit import fit_to_screen
    from .messages import titled
    from .metrics import set_margins

    dialog = Qt.QDialog(parent)
    dialog.setWindowTitle("Exercises")
    fit_to_screen(dialog, 720, 460)
    layout = Qt.QVBoxLayout(dialog)
    set_margins(layout, "dialog_margin")
    intro = Qt.QLabel("Guided exercises: each step says what to do in the app, with hints, the same step as code, and a "
                      "Check against your session. Packs come with plugins, or as folders.")
    intro.setWordWrap(True)
    layout.addWidget(intro)
    tree = Qt.QTreeWidget()
    tree.setHeaderLabels(["Exercise", "Level", "Minutes", "Audience"])
    tree.setRootIsDecorated(True)
    layout.addWidget(tree, 1)
    details = Qt.QLabel()
    details.setWordWrap(True)
    details.setTextFormat(QtNS.TextFormat.PlainText)
    layout.addWidget(details)
    dialog.packs, dialog.problems = [], []
    ROLE = int(QtNS.ItemDataRole.UserRole)

    def refresh() -> None:
        tree.clear()
        dialog.packs, dialog.problems = available_packs(settings)
        for pack in dialog.packs:
            top = Qt.QTreeWidgetItem([f"{pack.title}  ({pack.name} {pack.version}, "
                                      f"{'folder' if pack.source == 'folder' else 'plugin ' + pack.source})"
                                      + (f"  [{pack.classification}]" if pack.classification else ""), "", "", pack.audience])
            top.setData(0, ROLE, ("pack", pack))
            top.setToolTip(0, str(pack.path))
            for e in pack.exercises:
                item = Qt.QTreeWidgetItem([f"{e.number}. {e.title}", e.level, str(e.minutes or ""), ""])
                item.setData(0, ROLE, ("exercise", e))
                top.addChild(item)
            tree.addTopLevelItem(top)
            top.setExpanded(True)
        if not dialog.packs:
            empty = Qt.QTreeWidgetItem(["No exercise packs yet: Open folder… adds one."])
            empty.setFlags(QtNS.ItemFlag.NoItemFlags)
            tree.addTopLevelItem(empty)
        tree.resizeColumnToContents(0)
        details.setText("\n".join(f"Could not read {p}" for p in dialog.problems))
        update()

    def chosen() -> tuple[str, Any] | None:
        item = tree.currentItem()
        return item.data(0, ROLE) if item is not None else None

    def update() -> None:
        found = chosen()
        start_button.setEnabled(bool(found and found[0] == "exercise"))
        remove_button.setEnabled(bool(found and found[0] == "pack" and found[1].source == "folder"
                                      and str(found[1].path) in _folders(settings)))
        if found:
            pack = found[1] if found[0] == "pack" else found[1].pack
            lines = [pack.description] + ([f"For {pack.audience}."] if pack.audience else [])
            lines += [f"Classification: {pack.classification}"] if pack.classification else []
            lines += [pack.requirement_problem] if pack.requirement_problem else []
            if found[0] == "exercise" and found[1].data:
                lines.append(f"Data: {', '.join(found[1].data)}")
            if found[0] == "exercise" and found[1].models:
                lines.append(f"Models: {', '.join(pack.model_name(m) for m in found[1].models)}")
            details.setText("\n".join(lines))

    def add_folder(folder: str) -> Any:
        path = str(Path(folder).expanduser().resolve())
        pack = ex.load_pack(path)  # raises with the pack's problems: nothing is remembered
        mine = _folders(settings)
        if settings is not None and path not in mine:
            settings.setValue(FOLDERS_KEY, mine + [path])
        refresh()
        return pack

    def open_folder() -> None:
        folder = Qt.QFileDialog.getExistingDirectory(dialog, "Choose an exercise pack's folder")
        if folder:
            try:
                add_folder(folder)
            except Exception as exc:  # noqa: BLE001 - the pack's problems, shown
                Qt.QMessageBox.critical(dialog, *titled("Could not open the exercise pack", str(exc)))

    def remove() -> None:
        found = chosen()
        if found and found[0] == "pack" and settings is not None:
            settings.setValue(FOLDERS_KEY, [f for f in _folders(settings) if f != str(found[1].path)])
            refresh()

    def start_chosen() -> None:
        found = chosen()
        if found and found[0] == "exercise":
            dialog.accept()
            if start is not None:
                start(found[1])

    buttons = Qt.QHBoxLayout()
    open_button = Qt.QPushButton("Open folder…")
    open_button.setToolTip("add an exercise pack from a folder (it has pack.toml and exercises/)")
    open_button.clicked.connect(open_folder)
    remove_button = Qt.QPushButton("Remove folder")
    remove_button.clicked.connect(remove)
    start_button = Qt.QPushButton("Start")
    start_button.setDefault(True)
    start_button.clicked.connect(start_chosen)
    close = Qt.QPushButton("Close")
    close.clicked.connect(dialog.reject)
    for b in (open_button, remove_button):
        buttons.addWidget(b)
    buttons.addStretch()
    buttons.addWidget(start_button)
    buttons.addWidget(close)
    layout.addLayout(buttons)
    tree.currentItemChanged.connect(lambda *_: update())
    tree.itemDoubleClicked.connect(lambda *_: start_chosen())
    refresh()
    dialog.tree, dialog.details, dialog.add_folder, dialog.refresh = tree, details, add_folder, refresh
    dialog.start_button, dialog.open_button, dialog.remove_button = start_button, open_button, remove_button
    return dialog


def install(window: Any, dock_class: type, view_menu: Any, help_menu: Any, settings: Any, app: Any) -> Any:
    """Help → Exercises… in the main window, and the Exercise panel, made the first time an exercise starts (docked
    on the right, with its toggle in View). ``window.exercise_pane()`` gives the panel, made if need be;
    ``window.start_exercise(exercise)`` opens one."""
    from PySide6.QtCore import Qt as QtNS

    made: dict[str, Any] = {}

    def pane() -> Any:
        if "dock" not in made:
            dock = dock_class("Exercise", window)
            dock.setObjectName("exercise")
            made["pane"] = create_pane(app)
            dock.setWidget(made["pane"])
            window.addDockWidget(QtNS.DockWidgetArea.RightDockWidgetArea, dock)
            view_menu.addAction(dock.toggleViewAction())
            made["dock"] = dock
        return made["pane"]

    def start(exercise: ex.Exercise) -> None:
        p = pane()
        made["dock"].show()
        made["dock"].raise_()
        p.start(exercise)

    action = help_menu.addAction("Exercises…")
    action.setToolTip("Guided exercises from exercise packs: steps in the app, hints, the code, and checks")
    action.triggered.connect(lambda: exercises_dialog(window, settings, start).exec())
    window.exercise_pane, window.start_exercise, window.exercises_action = pane, start, action
    return action
