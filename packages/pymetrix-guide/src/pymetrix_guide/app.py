"""The Guide panel in the desktop app: a case study one step at a time, through the app's own dialogs.

The panel docks on the right (View → Guide). It shows:

- a choice of the case studies in the loaded courses, the chosen one's summary, and its problems if a block could
  not be read;
- if the case study starts from a demo project that is not open, an offer to open it;
- its introduction first, a page at a time (the ``##`` sections before the first step: an overview, the scenario);
- the stages of the workflow, as a strip with the current one in bold, when the case study names them;
- the step: its narrative, why it is done and what to look for, what will tick it, and progress
  ("Stage: Judgement — step 8 of 19 · 7 done");
- **Open…** (the step's dialog, filled in: the trainee runs it, changed or not), **Show me** (runs the step as it
  is written), **Back**, **Next** (through the introduction, then the steps) and **Start over** (the introduction's
  first page, and the demo again);
- every step, ticked when its check passes. Ticks follow the history: Undo takes one back, Redo gives it again.

It works the app only through the panel interface (``AppInterface``): the steps run as the app runs any step,
recorded, undoable and in the script, so the guide keeps no record of its own. Qt is imported only here.
"""

from __future__ import annotations

from typing import Any

from . import case_studies
from .casestudy import CaseStudy, CaseStudyError, Step, build_command

__all__ = ["GuidePanel", "create_panel", "intro_markdown", "stage_strip", "step_markdown"]

# the app's spacing tokens (pymetrix.ui.metrics: named sizes on a 4-pixel grid), as the sizes they have at Normal text:
# a plugin uses pymetrix's public surface only, and the app's metrics module is not part of it
_SIZES = {"none": 0, "tight": 4, "normal": 8}


def _margins(layout: Any, left: str, top: str | None = None, right: str | None = None, bottom: str | None = None) -> None:
    """A layout's contents margins from token names, CSS-like (one for all four sides; four for left, top, right, bottom)."""
    top = left if top is None else top
    names = (left, top, left if right is None else right, top if bottom is None else bottom)
    layout.setContentsMargins(*(_SIZES[n] for n in names))


def _spacing(layout: Any, token: str) -> None:
    layout.setSpacing(_SIZES[token])


def step_markdown(study: CaseStudy, index: int) -> str:
    """The Markdown the panel shows for step ``index`` (``len(steps)`` is the closing page)."""
    if index >= len(study.steps):
        return "## Finished\n\n" + (study.closing or "That is the whole case study.")
    step = study.steps[index]
    parts = [f"## {index + 1}. {step.title}", step.text]
    if step.why:
        parts.append(f"**Why this step.** {step.why}")
    if step.look:
        parts.append(f"**What to look for.** {step.look}")
    return "\n\n".join(p for p in parts if p).rstrip()


def intro_markdown(study: CaseStudy, page: int) -> str:
    """The Markdown the panel shows for page ``page`` of the introduction."""
    intro = study.intro[page]
    return f"## {intro.title}\n\n{intro.text}".rstrip()


def stage_strip(study: CaseStudy, current: str = "", finished: bool = False) -> str:
    """The workflow's stages as rich text: the current one in bold, those still to come in grey ("" without stages).
    On the introduction (no ``current``) and the finished page all are plain."""
    import html

    if not study.stages:
        return ""
    at = study.stages.index(current) if current in study.stages else (len(study.stages) if finished else -1)
    out = []
    for i, name in enumerate(study.stages):
        text = html.escape(name).replace(" ", "&nbsp;")
        if i == at:
            out.append(f"<b>{text}</b>")
        elif 0 <= at < i:
            out.append(f'<span style="color: #8a8a8a">{text}</span>')
        else:
            out.append(text)
    return "Workflow: " + " › ".join(out)


def _make_panel_class() -> type:
    from PySide6 import QtCore, QtWidgets

    Qt = QtCore.Qt

    class GuidePanel(QtWidgets.QWidget):
        """The Guide: one case study, one step at a time."""

        def __init__(self, app: Any) -> None:
            super().__init__()
            self.app = app
            self.studies: list[CaseStudy] = []
            self.study: CaseStudy | None = None
            self.index = 0  # the step (len(steps): the finished page)
            self.page: int | None = None  # the introduction's page shown, or None for a step
            self._positions: dict[str, tuple[int | None, int]] = {}
            self._courses_seen: tuple[str, ...] = ()
            outer = QtWidgets.QVBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)
            # the case study's choice at the top and the step's buttons at the foot stay in view; what is between
            # them (the step's text, its check, the list of steps) scrolls once the dock is smaller than its contents
            top = QtWidgets.QHBoxLayout()
            _margins(top, "normal", "normal", "normal", "none")
            self.choice = QtWidgets.QComboBox(objectName="guide-choice")
            self.choice.setToolTip("The case studies in the loaded course material (Help → Course material…)")
            self.choice.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            self.choice.setMinimumContentsLength(12)  # a long title shortens: the reload button stays in view
            self.choice.currentIndexChanged.connect(self._chosen)
            rescan = QtWidgets.QToolButton(objectName="guide-rescan", text="↻")
            rescan.setToolTip("Look again for case studies (after adding course material)")
            rescan.setAccessibleName("Look again for case studies")
            rescan.clicked.connect(self.reload_studies)
            top.addWidget(self.choice, 1)
            top.addWidget(rescan)
            outer.addLayout(top)
            self.scroll = QtWidgets.QScrollArea(objectName="guide-scroll", widgetResizable=True)
            self.scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            body = QtWidgets.QWidget(objectName="guide-body")
            self.scroll.setWidget(body)
            outer.addWidget(self.scroll, 1)
            layout = QtWidgets.QVBoxLayout(body)
            _margins(layout, "normal")

            self.summary = QtWidgets.QLabel(objectName="guide-summary", wordWrap=True)
            self.summary.setStyleSheet("font-style: italic")
            layout.addWidget(self.summary)

            self.problems = QtWidgets.QLabel(objectName="guide-problems", wordWrap=True)
            self.problems.setStyleSheet("color: #b3261e")
            layout.addWidget(self.problems)

            self.start_bar = QtWidgets.QFrame(objectName="guide-start")
            bar = QtWidgets.QHBoxLayout(self.start_bar)
            bar.setContentsMargins(0, 0, 0, 0)
            self.start_text = QtWidgets.QLabel(wordWrap=True)
            self.demo_button = QtWidgets.QPushButton("Open the demo", objectName="guide-demo")
            self.demo_button.clicked.connect(self.open_demo)
            bar.addWidget(self.start_text, 1)
            bar.addWidget(self.demo_button)
            layout.addWidget(self.start_bar)

            self.stages = QtWidgets.QLabel(objectName="guide-stages", wordWrap=True)
            self.stages.setTextFormat(Qt.TextFormat.RichText)
            self.stages.setToolTip("The stages of the workflow: where this step is in it")
            layout.addWidget(self.stages)
            self.progress = QtWidgets.QLabel(objectName="guide-progress", wordWrap=True)
            layout.addWidget(self.progress)
            self.text = QtWidgets.QTextBrowser(objectName="guide-text")
            self.text.setOpenExternalLinks(True)
            layout.addWidget(self.text, 3)
            self.check_label = QtWidgets.QLabel(objectName="guide-check", wordWrap=True)
            layout.addWidget(self.check_label)

            row = QtWidgets.QHBoxLayout()
            self.open_button = QtWidgets.QPushButton("Open…", objectName="guide-open")
            self.open_button.setToolTip("Open the step's dialog filled in: run it as it is, or change it first")
            self.show_button = QtWidgets.QPushButton("Show me", objectName="guide-show")
            self.show_button.setToolTip("Run the step as the case study writes it (a recorded step, as any other)")
            self.open_button.clicked.connect(self.open_step)
            self.show_button.clicked.connect(self.show_me)
            row.addWidget(self.open_button)
            row.addWidget(self.show_button)
            self.buttons = QtWidgets.QWidget(objectName="guide-buttons")  # pinned under the scrolling part
            pinned = QtWidgets.QVBoxLayout(self.buttons)
            _margins(pinned, "normal", "tight", "normal", "normal")
            _spacing(pinned, "tight")
            pinned.addLayout(row)
            nav = QtWidgets.QHBoxLayout()
            self.back_button = QtWidgets.QPushButton("Back", objectName="guide-back")
            self.next_button = QtWidgets.QPushButton("Next", objectName="guide-next")
            self.restart_button = QtWidgets.QPushButton("Start over", objectName="guide-restart")
            self.restart_button.setToolTip("Back to the introduction, and the case study's demo project as it first was")
            self.back_button.clicked.connect(self.back)
            self.next_button.clicked.connect(self.forward)
            self.restart_button.clicked.connect(self.start_over)
            for b in (self.back_button, self.next_button, self.restart_button):
                nav.addWidget(b)
            pinned.addLayout(nav)
            self._nav, self._fold = nav, QtWidgets.QHBoxLayout()  # Start over, on a row of its own when narrow
            pinned.addLayout(self._fold)

            self.steps = QtWidgets.QListWidget(objectName="guide-steps")
            self.steps.itemClicked.connect(lambda item: self.go(self.steps.row(item)))
            layout.addWidget(self.steps, 2)
            # the buttons' strip keeps its height; a dock narrower than the buttons scrolls them sideways
            self.button_strip = QtWidgets.QScrollArea(objectName="guide-button-strip", widgetResizable=True)
            self.button_strip.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            self.button_strip.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.button_strip.setWidget(self.buttons)
            outer.addWidget(self.button_strip)
            self._fit_strip()

            app.on_history_changed(self.refresh)
            self.reload_studies()

        def _fold_buttons(self, width: int) -> None:
            """Back, Next and Start over side by side where ``width`` has room for them, else Start over on a row of
            its own under Back and Next: the buttons need less width (wider buttons, as with Windows' fonts and style,
            no longer make a dock of ordinary width scroll them sideways)."""
            margins = self.buttons.layout().contentsMargins()
            row = (self.back_button, self.next_button, self.restart_button)
            spacing = self._nav.spacing()
            if spacing < 0:  # the style spaces by the kind of widget (as on Windows)
                button = QtWidgets.QSizePolicy.ControlType.PushButton
                spacing = self.style().layoutSpacing(button, button, QtCore.Qt.Orientation.Horizontal)
            three = sum(b.minimumSizeHint().width() for b in row) + 2 * max(spacing, 0) + margins.left() + margins.right()
            fold = width < three
            if fold and self._fold.indexOf(self.restart_button) < 0:
                self._nav.removeWidget(self.restart_button)
                self._fold.addWidget(self.restart_button)
            elif not fold and self._nav.indexOf(self.restart_button) < 0:
                self._fold.removeWidget(self.restart_button)
                self._nav.addWidget(self.restart_button)

        def _fit_strip(self) -> None:
            """The buttons' strip as tall as the buttons, and the sideways scroll bar when it shows."""
            strip = self.button_strip
            self._fold_buttons(self.width())  # the strip is as wide as the panel
            self.buttons.layout().activate()
            bar = strip.horizontalScrollBar()
            needs_bar = self.buttons.minimumSizeHint().width() > strip.viewport().width() > 0
            strip.setFixedHeight(self.buttons.sizeHint().height() + (bar.sizeHint().height() if needs_bar else 0))

        def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt's name
            super().resizeEvent(event)
            self._fit_strip()

        # -- the case studies ------------------------------------------------------------------------------------

        def _session(self) -> Any:
            try:
                return self.app.session
            except Exception:  # noqa: BLE001 - no session yet
                return None

        def reload_studies(self) -> None:
            current = self.study.path if self.study is not None else None
            try:
                self.studies = case_studies(self._session())
            except Exception as exc:  # noqa: BLE001 - a bad folder must not stop the panel
                self.studies = []
                self.app.status(f"Guide: could not read the case studies: {exc}")
            self._courses_seen = tuple(getattr(self._session(), "courses", []) or [])
            self.choice.blockSignals(True)
            self.choice.clear()
            for study in self.studies:
                self.choice.addItem(study.title + (f" · {study.level}" if study.level else "")
                                    + (" (with problems)" if study.problems else ""))
                if study.summary:
                    self.choice.setItemData(self.choice.count() - 1, study.summary, Qt.ItemDataRole.ToolTipRole)
            self.choice.blockSignals(False)
            if not self.studies:
                self.choice.addItem("No case studies: add course material (Help → Course material…)")
                self.choice.setEnabled(False)
                self.study = None
            else:
                self.choice.setEnabled(True)
                at = next((i for i, s in enumerate(self.studies) if s.path == current), 0)
                self.choice.setCurrentIndex(at)
                self.study = self.studies[at]
                self._restore()
            self.refresh()

        def _start(self) -> tuple[int | None, int]:
            return (0 if self.study is not None and self.study.intro else None), 0

        def _restore(self) -> None:
            self.page, self.index = self._positions.get(str(self.study.path), self._start())
            if (self.page is not None and self.page >= len(self.study.intro)) or self.index > len(self.study.steps):
                self.page, self.index = self._start()  # the case study changed since

        def _remember(self) -> None:
            if self.study is not None:
                self._positions[str(self.study.path)] = (self.page, self.index)

        def choose(self, title: str) -> None:
            """Choose a case study by its title."""
            at = next(i for i, s in enumerate(self.studies) if s.title == title)
            self.choice.setCurrentIndex(at)

        def _chosen(self, at: int) -> None:
            if 0 <= at < len(self.studies):
                self._remember()
                self.study = self.studies[at]
                self._restore()
                self.refresh()

        # -- the steps ---------------------------------------------------------------------------------------------

        def step(self) -> Step | None:
            if self.study is None or self.page is not None or self.index >= len(self.study.steps):
                return None
            return self.study.steps[self.index]

        def done(self) -> list[bool]:
            """Whether each step's check passes now (a step with no check never ticks itself)."""
            s = self._session()
            return [bool(st.check is not None and st.check.holds(s)) for st in (self.study.steps if self.study else ())]

        def go(self, index: int) -> None:
            """Show step ``index`` (``len(steps)``: the finished page)."""
            if self.study is None:
                return
            self.page = None
            self.index = max(0, min(index, len(self.study.steps)))
            self._remember()
            self.refresh()

        def go_intro(self, page: int = 0) -> None:
            """Show page ``page`` of the introduction (step 1 when there is none)."""
            if self.study is None:
                return
            if not self.study.intro:
                self.go(0)
                return
            self.page = max(0, min(page, len(self.study.intro) - 1))
            self.index = 0
            self._remember()
            self.refresh()

        def back(self) -> None:
            if self.study is None:
                return
            if self.page is not None:
                self.go_intro(self.page - 1)
            elif self.index == 0 and self.study.intro:
                self.go_intro(len(self.study.intro) - 1)
            else:
                self.go(self.index - 1)

        def forward(self) -> None:
            if self.study is None:
                return
            if self.page is not None and self.page + 1 < len(self.study.intro):
                self.go_intro(self.page + 1)
            elif self.page is not None:
                self.go(0)
            else:
                self.go(self.index + 1)

        def command(self) -> Any:
            step = self.step()
            return build_command(step.command, step.settings) if step is not None and step.command else None

        def open_step(self) -> None:
            try:
                cmd = self.command()
            except CaseStudyError as exc:
                self.app.status(f"Guide: {exc}")
                return
            if cmd is not None:
                self.app.open_dialog(type(cmd), cmd)
                self.refresh()

        def show_me(self) -> None:
            try:
                cmd = self.command()
                if cmd is None:
                    return
                self.app.run(cmd)
            except Exception as exc:  # noqa: BLE001 - the step's error is shown, as the app shows it
                self.app.status(f"Guide: the step did not run: {exc}")
                self.check_label.setText(f"The step did not run: {exc}")
                return
            self.refresh()

        # -- the demo it starts from -----------------------------------------------------------------------------------

        def needs_demo(self) -> bool:
            return bool(self.study and self.study.demo and getattr(self._session(), "demo", None) != self.study.demo)

        def open_demo(self) -> None:
            """Open the case study's demo project (as it first was), through the app; the app asks to save the work."""
            if self.study is None or not self.study.demo:
                return
            from pymetrix.errors import NotSupportedError

            opener = getattr(self.app, "open_demo", None)  # extension API 1.7; an app before it has none
            try:
                if not callable(opener):
                    raise NotSupportedError("open_demo")
                opener(self.study.demo)  # the app asks to save unsaved work; the trainee may cancel
            except NotSupportedError:  # say where it is in the menus instead
                again = getattr(self._session(), "demo", None) == self.study.demo
                self.app.status(f"Guide: open the demo with File → {'Reload demo project' if again else 'Open demo project…'} "
                                f"({self.study.demo})")
            except Exception as exc:  # noqa: BLE001 - e.g. a demo this pymetrix does not have: said, not raised
                self.app.status(f"Guide: the demo {self.study.demo} did not open: {exc}")
            self.refresh()

        def start_over(self) -> None:
            self.go_intro(0)
            if self.study is not None and self.study.demo:
                self.open_demo()
            else:
                self.app.status("Guide: back to the start (File → New session starts from an empty workspace)")

        # -- showing it ------------------------------------------------------------------------------------------------

        def refresh(self) -> None:
            import shiboken6

            if not (shiboken6.isValid(self) and shiboken6.isValid(self.steps)):
                return  # the panel, or its window, is gone (an app older than 1.7 may still call it): nothing to show
            if tuple(getattr(self._session(), "courses", []) or []) != self._courses_seen:
                self.reload_studies()  # another project, with its own course material
                return
            study = self.study
            ticks = self.done()
            self.summary.setVisible(bool(study and study.summary))
            self.summary.setText(study.summary if study else "")
            self.problems.setVisible(bool(study and study.problems))
            if study and study.problems:
                self.problems.setText("Some of this case study could not be read:\n" + "\n".join(study.problems))
            self.start_bar.setVisible(self.needs_demo())
            if study is not None and study.demo:
                self.start_text.setText(f"This case study starts from the demo project <b>{study.demo}</b>.")
            self.steps.clear()
            for i, st in enumerate(study.steps if study else ()):
                item = QtWidgets.QListWidgetItem(f"{i + 1}. {st.title}")
                if st.stage:
                    item.setToolTip(f"Stage: {st.stage}")
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                item.setCheckState(Qt.CheckState.Checked if ticks[i] else Qt.CheckState.Unchecked)
                self.steps.addItem(item)
            if study is None:
                self.stages.setVisible(False)
                self.progress.setText("")
                self.text.setMarkdown("")
                self.check_label.setText("")
                for b in (self.open_button, self.show_button, self.back_button, self.next_button, self.restart_button):
                    b.setEnabled(False)
                return
            n = len(study.steps)
            if study.path is not None:
                self.text.document().setBaseUrl(QtCore.QUrl.fromLocalFile(str(study.path.parent) + "/"))
            step = self.step()
            if self.page is not None:
                self.text.setMarkdown(intro_markdown(study, self.page))
                m = len(study.intro)
                self.progress.setText(f"Introduction, page {self.page + 1} of {m} · then {n} steps · {sum(ticks)} done")
                self.check_label.setText(f"Next: step 1, {study.steps[0].title}." if n else "")
                self.steps.setCurrentRow(-1)
            elif step is None:
                self.text.setMarkdown(step_markdown(study, self.index))
                self.progress.setText(f"{study.title}: finished · {sum(ticks)} of {n} steps done")
                self.check_label.setText("")
            else:
                self.text.setMarkdown(step_markdown(study, self.index))
                where = f"step {self.index + 1} of {n} · {sum(ticks)} done"
                self.progress.setText(f"Stage: {step.stage} — {where}" if step.stage else where[0].upper() + where[1:])
                self.steps.setCurrentRow(self.index)
                if step.check is None:
                    self.check_label.setText("No check for this step: press Next when you are ready.")
                elif ticks[self.index]:
                    self.check_label.setText(f"✓ Done: {step.check.describe()}.")
                else:
                    self.check_label.setText(f"To do: {step.check.describe()}.")
            strip = stage_strip(study, step.stage if step is not None else "", finished=self.page is None and step is None)
            self.stages.setVisible(bool(strip))
            self.stages.setText(strip)
            has_command = step is not None and bool(step.command)
            self.open_button.setEnabled(has_command)
            self.show_button.setEnabled(has_command)
            self.back_button.setEnabled(bool(self.page) if self.page is not None else (self.index > 0 or bool(study.intro)))
            self.next_button.setEnabled(self.page is not None or self.index < n)
            self.restart_button.setEnabled(True)

    return GuidePanel


_PANEL: type | None = None


def create_panel(app: Any) -> Any:
    """The Guide panel (a QWidget) for the app's ``AppInterface``."""
    global _PANEL
    if _PANEL is None:
        _PANEL = _make_panel_class()
    return _PANEL(app)


def __getattr__(name: str) -> Any:
    if name == "GuidePanel":
        global _PANEL
        if _PANEL is None:
            _PANEL = _make_panel_class()
        return _PANEL
    raise AttributeError(name)
