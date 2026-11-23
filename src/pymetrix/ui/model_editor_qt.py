"""Model > Model editor: a model written as text, checked as it is typed (pymetrix.models.source): pymetrix's syntax,
Dynare's, IRIS's or EViews's (a model's text or a .prg, pymetrix.models.eviews), chosen in the syntax box.

The text on the left; on the right, the Problems (each at its line and column: click one to go there) and the Symbols
(each name's role, value, description and the lines that use it). Problems are underlined where they are, red for
errors and amber for warnings; hovering shows the message. Typing two characters of a word (or Ctrl+Space) offers
the names the text declares, the functions and the keywords; hovering a name shows its role, value, description,
estimation settings and the lines that use it, and hovering a function its signature and what it does. The Equations tab shows the equations in algebraic notation
(Copy LaTeX for a paper); Check solution finds the steady state and the roots, and says whether the Blanchard-Kahn
conditions hold; when the solution is unique, the Solution tab also draws the impulse responses (``SolutionCheck.irf``:
one-standard-deviation shocks, correlated ones orthogonalised as ``LinearSolution.irf``) to the shock chosen, for
the variables ticked (the first six endogenous by default), 20 periods ahead; with regimes, the base regime's. Apply builds the model into the session as a step (``DefineModel``), which keeps the text as
written."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from .editor_qt import CodeEditor, _dark, _format
from . import theme
from .metrics import add_space

_PYMETRIX_WORDS = ("parameters", "shocks", "exogenous", "variables", "equations", "measurement", "reported", "regimes", "error",
                   "from", "corr", "surprise")
_DYNARE_WORDS = ("var", "varexo", "varexo_det", "parameters", "model", "end", "shocks", "initval", "endval", "steady_state_model",
                 "estimated_params", "varobs", "estimation", "stoch_simul", "steady", "check", "stderr", "corr", "histval")
TEMPLATES = {
    "pymetrix": "# a model in pymetrix's syntax\nparameters\n    rho = 0.8\nshocks\n    e = 1\nequations\n    x = rho*x[-1] + e\n",
    "dynare": "// a model in Dynare's syntax\nvar x;\nvarexo e;\nparameters rho;\nrho = 0.8;\nmodel;\n  x = rho*x(-1) + e;\nend;\n",
    "iris": "% a model in IRIS's syntax\n!transition_variables\n    x\n!transition_shocks\n    e\n!parameters\n    rho = 0.8, std_e = 1\n"
            "!transition_equations\n    x = rho*x{-1} + e;\n",
    "eviews": "' a model in EViews's syntax (a program: model, append, coefficients)\nc(1) = 0.8\nmodel m\n"
              "m.append x = c(1)*x(-1)\nm.append @innov x 1\n",
}
LABELS = {"pymetrix": "pymetrix", "dynare": "Dynare (.mod)", "iris": "IRIS (.model)", "eviews": "EViews (.prg / model text)"}
_EVIEWS_WORDS = ("model", "equation", "append", "merge", "scalar", "coef", "vector", "param", "smpl", "series", "ls")


class ModelHighlighter(QtGui.QSyntaxHighlighter):
    """Section keywords (or Dynare's), comments, numbers, descriptions, leads and lags."""

    def __init__(self, document: QtGui.QTextDocument, syntax: str = "pymetrix", dark: bool = False) -> None:
        super().__init__(document)
        self.dark = dark
        self.set_syntax(syntax)

    def set_syntax(self, syntax: str) -> None:
        self.syntax = syntax
        words = {"pymetrix": _PYMETRIX_WORDS, "eviews": _EVIEWS_WORDS}.get(syntax, _DYNARE_WORDS)
        c = theme.MODEL_SYNTAX["dark" if self.dark else "light"]
        keyword = _format(c["keyword"], bold=True)
        number = _format(c["number"])
        string = _format(c["string"])
        timing = _format(c["timing"])
        comment = _format(c["comment"], italic=True)
        comments = {"pymetrix": r"#.*$", "dynare": r"(//|%).*$", "iris": r"%.*$", "eviews": r"'.*$"}[syntax]
        keywords = (re.compile(r"![A-Za-z_\-]+") if syntax == "iris" else
                    re.compile(rf"(?i)@[A-Za-z_]\w*|\b({'|'.join(words)})\b") if syntax == "eviews" else
                    re.compile(rf"\b({'|'.join(words)})\b"))
        times = {"pymetrix": r"\[[+-]?\d+\]", "dynare": r"\([+-]\d+\)", "iris": r"\{[+-]?\d+\}", "eviews": r"\(\s*[+-]?\d+\s*\)"}[syntax]
        strings = r"\"[^\"]*\"" if syntax == "eviews" else r"\"[^\"]*\"|'[^']*'"  # EViews: ' starts a comment
        self.rules = [(keywords, keyword), (re.compile(r"\b\d+(\.\d*)?([eE][+-]?\d+)?\b"), number), (re.compile(times), timing),
                      (re.compile(strings), string), (re.compile(comments), comment)]
        self.rehighlight()

    def highlightBlock(self, text: str) -> None:  # noqa: N802 (Qt override)
        for pattern, style in self.rules:
            for m in pattern.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), style)


class ModelTextEditor(CodeEditor):
    """The code editor with a model's highlighting, problems underlined where they are (not whole lines), completion
    of the names the text declares, the functions and the keywords (after two characters, or Ctrl+Space), and hover
    help on names and functions (``ModelSource.completions`` and ``hover``, from the window's last check)."""

    def __init__(self, syntax: str = "pymetrix", parent: Any = None) -> None:
        super().__init__(None, parent)
        self.highlighter.setDocument(None)
        self.highlighter = ModelHighlighter(self.document(), syntax, _dark(self))
        self.diagnostics: list[Any] = []
        self.syntax = syntax
        self.found: Any = None  # the last check (SourceCheck) of the text, for completions and hover help

    def _source(self) -> Any:
        from ..models.source import ModelSource

        return ModelSource(self.toPlainText(), self.syntax)

    def completions(self) -> list[str]:
        """What may complete the word before the cursor (``ModelSource.completions``)."""
        line, column = self._line_col()
        try:
            return self._source().completions(line, column + 1, self.found)
        except Exception:  # noqa: BLE001 - completion is a convenience; never let it break typing
            return []

    def _word_before(self) -> str:
        before = self.textCursor().block().text()[: self.textCursor().positionInBlock()]
        m = re.search(r"!?[A-Za-z_][\w\-]*$" if self.syntax == "iris" else r"@?[A-Za-z_]\w*$" if self.syntax == "eviews" else r"[A-Za-z_]\w*$", before)
        return m.group(0) if m else ""

    def complete(self, force: bool = False) -> None:
        """Show the completions: when two characters of a word are typed, or at once with ``force`` (Ctrl+Space)."""
        self._prefix = self._word_before()
        if not force and len(self._prefix) < 2:
            self.completer.popup().hide()
            return
        names = self.completions()
        if not names:
            self.completer.popup().hide()
            return
        self.completer.setModel(QtCore.QStringListModel(names, self.completer))
        self.completer.setCompletionPrefix(self._prefix)
        if self.completer.completionCount() == 0:
            self.completer.popup().hide()
            return
        rect = self.cursorRect()
        rect.setWidth(self.completer.popup().sizeHintForColumn(0) + self.completer.popup().verticalScrollBar().sizeHint().width() + 8)
        self.completer.complete(rect)

    def _insert_completion(self, name: str) -> None:
        cursor = self.textCursor()  # the word typed so far replaced (completion ignores case)
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.Left, QtGui.QTextCursor.MoveMode.KeepAnchor, len(self._word_before()))
        cursor.insertText(name)
        self.setTextCursor(cursor)

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:  # noqa: N802
        visible = self.completer.popup().isVisible()
        super().keyPressEvent(e)
        text = e.text()
        if not visible and text and (text.isalnum() or text in "_!"):
            self.complete()

    def signature(self) -> str:
        """The function before the bracket just typed: its signature and what it does."""
        from ..models.source import function_help

        before = self.textCursor().block().text()[: self.textCursor().positionInBlock()]
        m = re.search(r"([A-Za-z_]\w*)\s*\($", before)
        return function_help(m.group(1)) if m else ""

    def help_at(self, line: int, column: int) -> str:
        """The hover text at ``line`` and ``column`` (from 1): an underlined problem's message, else what the name or
        function there is (``ModelSource.hover``), else the line's first problem."""
        for d in self.diagnostics:
            if d.line == line and d.severity in ("error", "warning") and d.column <= column < d.column + max(d.length, 1):
                return d.message
        try:
            text = self._source().hover(line, column, self.found)
        except Exception:  # noqa: BLE001 - help is a convenience
            text = ""
        return text or self.problems.get(line, "")

    def event(self, e: QtCore.QEvent) -> bool:
        if e.type() == QtCore.QEvent.Type.ToolTip:
            cursor = self.cursorForPosition(self.viewport().mapFromGlobal(e.globalPos()))
            line, at = cursor.blockNumber() + 1, cursor.positionInBlock()
            text = self.help_at(line, at + 1) or (self.help_at(line, at) if at else "")
            if text:
                QtWidgets.QToolTip.showText(e.globalPos(), text, self)
            else:
                QtWidgets.QToolTip.hideText()
            return True
        return super().event(e)

    def set_diagnostics(self, diagnostics: list[Any]) -> None:
        self.diagnostics = list(diagnostics)
        self.problems = {}
        for d in self.diagnostics:  # line numbers in red, and the message on hover
            if d.severity in ("error", "warning"):
                self.problems.setdefault(d.line, d.message)
        self._decorate()
        self.numbers.update()

    def _decorate(self) -> None:
        dark = _dark(self)
        line = QtWidgets.QTextEdit.ExtraSelection()
        line.format.setBackground(QtGui.QColor(theme.EDITOR_BACKGROUNDS["current line"][0 if dark else 1]))
        line.format.setProperty(QtGui.QTextFormat.Property.FullWidthSelection, True)
        line.cursor = self.textCursor()
        line.cursor.clearSelection()
        selections = [line]
        for d in getattr(self, "diagnostics", []):
            if d.severity not in ("error", "warning"):
                continue
            block = self.document().findBlockByNumber(d.line - 1)
            if not block.isValid():
                continue
            mark = QtWidgets.QTextEdit.ExtraSelection()
            mark.format.setUnderlineStyle(QtGui.QTextCharFormat.UnderlineStyle.WaveUnderline)
            mark.format.setUnderlineColor(theme.colour("error" if d.severity == "error" else "warning"))
            start = min(max(d.column - 1, 0), max(block.length() - 1, 0))
            cursor = QtGui.QTextCursor(block)
            cursor.setPosition(block.position() + start)
            end = min(block.position() + start + max(d.length, 1), block.position() + max(block.length() - 1, 0))
            cursor.setPosition(max(end, block.position() + start), QtGui.QTextCursor.MoveMode.KeepAnchor)
            mark.cursor = cursor
            selections.append(mark)
        selections += self._bracket_match()
        self.setExtraSelections(selections)

    def check(self) -> dict[int, str]:  # the model is checked by the window, not pyflakes
        return self.problems


class ModelEditorWindow(QtWidgets.QWidget):
    """A model as text with its checks: see the module docstring. ``apply(command)`` records the step (the main
    window's), returning whether it worked."""

    def __init__(self, apply: Callable[[Any], bool] | None = None, text: str = "", syntax: str = "pymetrix", name: str = "model",
                 parent: Any = None, open_other: Callable[..., Any] | None = None, translation: Any = None) -> None:
        super().__init__(parent)
        self.apply_step = apply
        self.open_other = open_other  # opens another editor (an export); a window of its own when not given
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QtWidgets.QToolBar()
        self.syntax = QtWidgets.QComboBox()
        for key, label in LABELS.items():
            self.syntax.addItem(label, key)
        self.syntax.setCurrentIndex(max(0, self.syntax.findData(syntax)))
        self.name = QtWidgets.QLineEdit(name)
        self.name.setMaximumWidth(160)
        self.name.setToolTip("the name the model gets in the session")
        bar.addWidget(QtWidgets.QLabel(" Syntax "))
        bar.addWidget(self.syntax)
        bar.addWidget(QtWidgets.QLabel("  Name "))
        bar.addWidget(self.name)
        self.apply_action = bar.addAction("Apply", self.apply)
        self.apply_action.setToolTip("build the model into the session (a step that keeps this text)")
        export = QtWidgets.QToolButton()
        export.setText("Write as")
        export.setToolTip("the model in another syntax: a new editor with the text, and notes on what does not carry across")
        export.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QtWidgets.QMenu(export)
        for key, label in LABELS.items():
            menu.addAction(label, lambda k=key: self.export(k))
        export.setMenu(menu)
        self.export_button, self.export_menu = export, menu
        bar.addWidget(export)
        self.solve_action = bar.addAction("Check solution", self.check_solution)
        self.solve_action.setToolTip("find the steady state and the roots of the linearised model: is there a unique stable "
                                     "solution (Blanchard-Kahn)?")
        layout.addWidget(bar)
        split = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        self.editor = ModelTextEditor(syntax)
        self.editor.setPlainText(text or TEMPLATES[syntax])
        split.addWidget(self.editor)
        side = QtWidgets.QTabWidget()
        self.problems = QtWidgets.QTreeWidget()
        self.problems.setHeaderLabels(["Line", "Col", "", "Problem"])
        self.problems.setRootIsDecorated(False)
        self.problems.itemActivated.connect(self._go)
        self.problems.itemClicked.connect(self._go)
        self.symbols = QtWidgets.QTreeWidget()
        self.symbols.setHeaderLabels(["Symbol", "Role", "Value", "Description", "Used on lines", "Estimation"])
        self.symbols.setRootIsDecorated(False)
        self.symbols.itemActivated.connect(self._go)
        side.addTab(self.problems, "Problems")
        side.addTab(self.symbols, "Symbols")
        side.addTab(self._equations_tab(), "Equations")
        side.addTab(self._solution_tab(), "Solution")
        self.translation = translation
        if translation is not None:  # written from another syntax: what did not carry across, and the check
            notes = QtWidgets.QPlainTextEdit()
            notes.setReadOnly(True)
            check = (f"Read back and solved: responses and steady state within {translation.difference:.2g} of the original."
                     if translation.checked and translation.difference is not None else "Not checked against the original.")
            notes.setPlainText(check + "\n\n" + "\n".join(f"- {n}" for n in translation.notes))
            side.addTab(notes, "Translation")
            side.setCurrentWidget(notes)
        self.sidecars = side
        side.currentChanged.connect(lambda _i: self._show_equations())
        split.addWidget(side)
        split.setSizes([620, 380])
        self.split, self.split_moved = split, False
        split.splitterMoved.connect(lambda *_: setattr(self, "split_moved", True))
        layout.addWidget(split, 1)
        self.status = QtWidgets.QLabel()
        layout.addWidget(self.status)
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(300)  # checked a moment after typing stops
        self.timer.timeout.connect(self.refresh)
        self.editor.textChanged.connect(self.timer.start)
        self.syntax.currentIndexChanged.connect(self._syntax_changed)
        self.last = None
        self.equations_stale = True
        self.refresh()

    SIDE_SHARE = 0.38  # of the width, for the tabs beside the text

    def resizeEvent(self, event: Any) -> None:
        """The tabs beside the text get their share of the width, and room for every tab's name when the window has
        it, until the divider is dragged."""
        super().resizeEvent(event)
        if self.split_moved:
            return
        total = sum(self.split.sizes())
        if total <= 0:
            return
        want = max(int(total * self.SIDE_SHARE), self.sidecars.tabBar().sizeHint().width() + 4)
        want = min(want, total // 2)
        self.split.setSizes([total - want, want])

    def _equations_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        self.equations = QtWidgets.QTreeWidget()
        self.equations.setHeaderLabels(["Equation", "In algebraic notation"])
        self.equations.setRootIsDecorated(False)
        font = self.equations.font()
        font.setPointSizeF(font.pointSizeF() * 1.15)
        self.equations.setFont(font)
        self.equations.itemActivated.connect(self._go)
        self.equations.itemClicked.connect(self._go)
        self.equations_note = QtWidgets.QLabel()
        self.equations_note.setWordWrap(True)
        copy = QtWidgets.QPushButton("Copy LaTeX")
        copy.setToolTip("copy the equations as LaTeX align blocks, each tagged with its name")
        copy.clicked.connect(self.copy_latex)
        lay.addWidget(self.equations, 1)
        lay.addWidget(self.equations_note)
        lay.addWidget(copy, 0, Qt.AlignmentFlag.AlignRight)
        return page

    def _solution_tab(self) -> QtWidgets.QWidget:
        from .fit import scrolled_page

        page, body = scrolled_page()  # the verdict, steady state, roots and responses scroll in a short window
        lay = QtWidgets.QVBoxLayout(body)
        lay.setContentsMargins(0, 0, 0, 0)
        self.verdict = QtWidgets.QLabel("Check solution finds the steady state and the roots of the linearised model.")
        self.verdict.setWordWrap(True)
        self.steady = QtWidgets.QTreeWidget()
        self.steady.setHeaderLabels(["Variable", "Steady state", "Description"])
        self.steady.setRootIsDecorated(False)
        self.steady.itemActivated.connect(self._go)
        self.roots = QtWidgets.QTreeWidget()
        self.roots.setHeaderLabels(["", "Modulus", "Real", "Imaginary", ""])
        self.roots.setRootIsDecorated(False)
        lay.addWidget(self.verdict)
        lay.addWidget(self.steady, 2)
        lay.addWidget(self.roots, 1)
        add_space(lay)  # a section's room above a group box, as above a section's heading
        lay.addWidget(self._irf_box(), 3)
        self.solution_page = page
        return page

    IRF_PERIODS = 20
    IRF_VARIABLES = 6

    def _irf_box(self) -> QtWidgets.QWidget:
        """Impulse responses of the unique solution: a shock chooser, the variables to draw and the figure."""
        box = QtWidgets.QGroupBox("Impulse responses")
        lay = QtWidgets.QVBoxLayout(box)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Shock"))
        self.irf_shock = QtWidgets.QComboBox()
        self.irf_shock.setToolTip("the shock whose responses are drawn (one standard deviation; correlated shocks "
                                  "orthogonalised by Cholesky in the model's order)")
        row.addWidget(self.irf_shock, 1)
        lay.addLayout(row)
        self.irf_note = QtWidgets.QLabel("Drawn when Check solution finds a unique stable solution.")
        self.irf_note.setWordWrap(True)
        lay.addWidget(self.irf_note)
        split = QtWidgets.QHBoxLayout()
        self.irf_variables = QtWidgets.QListWidget()
        self.irf_variables.setToolTip("the variables drawn: tick or untick")
        self.irf_variables.setMaximumWidth(140)
        split.addWidget(self.irf_variables)
        self.irf_canvas = None
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self.irf_canvas = FigureCanvasQTAgg(Figure(figsize=(4, 3), layout="constrained"))
            self.irf_canvas.setMinimumSize(240, 180)
            split.addWidget(self.irf_canvas, 1)
        except ImportError:
            split.addWidget(QtWidgets.QLabel("Figures need matplotlib: pip install matplotlib"), 1)
        lay.addLayout(split, 1)
        self.irf = None
        self.irf_shock.currentIndexChanged.connect(lambda _i: self.draw_irf())
        self.irf_variables.itemChanged.connect(lambda _item: self.draw_irf())
        return box

    def _set_irf(self, found: Any) -> None:
        """The responses of a new check (or none), with its shocks and variables in the choosers."""
        from ..errors import PyMetrixError

        self.irf = None
        if found is not None and found.ok:
            try:
                self.irf = found.irf(self.IRF_PERIODS)
            except (PyMetrixError, ValueError, ArithmeticError) as exc:
                self.irf_note.setText(f"The responses could not be computed: {exc}")
        previous = self.irf_shock.currentText()
        ticked = {self.irf_variables.item(i).text() for i in range(self.irf_variables.count())
                  if self.irf_variables.item(i).checkState() == Qt.CheckState.Checked}
        known = {self.irf_variables.item(i).text() for i in range(self.irf_variables.count())}
        for widget in (self.irf_shock, self.irf_variables):
            widget.blockSignals(True)
        self.irf_shock.clear()
        self.irf_variables.clear()
        if self.irf is not None:
            self.irf_shock.addItems(list(self.irf.shocks))
            self.irf_shock.setCurrentIndex(max(0, self.irf_shock.findText(previous)))
            for k, name in enumerate(self.irf.responses):
                item = QtWidgets.QListWidgetItem(name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                on = name in ticked if known & set(self.irf.responses) else k < self.IRF_VARIABLES
                item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
                self.irf_variables.addItem(item)
            regimes = found.regimes
            self.irf_note.setText(f"{self.IRF_PERIODS} periods, one-standard-deviation shocks, deviations from the steady state"
                                  + (f"; the base regime's responses (the model has {regimes} regime{'s' if regimes != 1 else ''})"
                                     if regimes else ""))
        elif found is not None and not found.ok:
            self.irf_note.setText("No responses: the model has no unique stable solution.")
        for widget in (self.irf_shock, self.irf_variables):
            widget.blockSignals(False)
        self.draw_irf()

    def irf_shown(self) -> dict[str, Any]:
        """What the figure shows: each ticked variable's responses to the chosen shock (horizons 0..20)."""
        if self.irf is None or self.irf_shock.currentIndex() < 0:
            return {}
        shock = self.irf_shock.currentText()
        names = [self.irf_variables.item(i).text() for i in range(self.irf_variables.count())
                 if self.irf_variables.item(i).checkState() == Qt.CheckState.Checked]
        return {name: self.irf[name, shock].to_numpy() for name in names}

    def draw_irf(self) -> None:
        if self.irf_canvas is None:
            return
        from matplotlib import rc_context

        from ..figure_style import chart_rc, scaled
        from ..themes import get_theme
        from .chart_look import look

        fig = self.irf_canvas.figure
        fig.clear()
        shown = self.irf_shown()
        font, surface = look(self)
        house = scaled(get_theme(), surface.text_scale)  # the app's text size
        k = surface.text_scale
        with rc_context(chart_rc(house, font, surface)):  # the charts' house look, in the app's font and colours
            fig.set_facecolor(surface.background)
            if shown:
                n = len(shown)
                cols = 1 if n == 1 else 2
                rows = -(-n // cols)
                for i, (name, values) in enumerate(shown.items()):
                    ax = fig.add_subplot(rows, cols, i + 1)
                    ax.plot(range(len(values)), values, color=house.palette[0], linewidth=1.4)
                    ax.axhline(0, color=theme.CHART["zero line"], linewidth=0.6)
                    ax.set_title(name, fontsize=8 * k)
                    ax.tick_params(labelsize=7 * k)
                fig.suptitle(f"Responses to {self.irf_shock.currentText()}", fontsize=9 * k)
        self.irf_canvas.draw_idle()

    def _lines(self) -> dict[str, int]:
        """The line that determines or declares each name, from the last check."""
        return {s.name: s.line for s in (self.last.symbols if self.last is not None else ())}

    def _show_equations(self) -> None:
        """The Equations tab, built from the model when it is shown (building is too slow for every keystroke)."""
        if not self.equations_stale or self.sidecars.currentWidget() is not self.equations.parentWidget():
            return
        from ..errors import PyMetrixError

        self.equations_stale = False
        self.equations.clear()
        if self.last is None or not self.last.ok:
            self.equations_note.setText("The text has errors: see Problems.")
            return
        try:
            rows = self.source().algebraic()
        except (PyMetrixError, ValueError) as exc:
            self.equations_note.setText(f"Could not be shown: {exc}")
            return
        lines = self._lines()
        for eq in rows:
            item = QtWidgets.QTreeWidgetItem([eq.name if eq.kind == "equation" else f"{eq.name} (measurement)", eq.text])
            item.setData(0, Qt.ItemDataRole.UserRole, (lines.get(eq.variable, 0), 1))
            item.setToolTip(1, eq.text)
            self.equations.addTopLevelItem(item)
        self.equations.resizeColumnToContents(0)
        self.equations_note.setText("")

    def copy_latex(self) -> str:
        """The equations as LaTeX, on the clipboard (and returned); empty, with the reason, when they cannot be."""
        from ..errors import PyMetrixError

        try:
            text = self.source().latex()
        except (PyMetrixError, ValueError) as exc:
            self.status.setText(f" Could not write LaTeX: {exc}")
            return ""
        QtWidgets.QApplication.clipboard().setText(text)
        self.status.setText(" The equations are on the clipboard as LaTeX")
        return text

    def check_solution(self) -> Any:
        """Find the steady state and the roots (``ModelSource.check_solution``) and show them in the Solution tab."""
        from ..errors import PyMetrixError

        self.refresh()
        self.sidecars.setCurrentWidget(self.solution_page)
        self.steady.clear()
        self.roots.clear()
        if self.last is None or not self.last.ok:
            self._verdict("The text has errors: see Problems.", "error")
            self._set_irf(None)
            return None
        try:
            found = self.source().check_solution()
        except (PyMetrixError, ValueError) as exc:
            self._verdict(f"The model could not be built: {exc}", "error")
            self._set_irf(None)
            return None
        self._verdict(found.message, "ok" if found.ok else "error")
        lines = self._lines()
        for name, value in (found.steady or {}).items():
            item = QtWidgets.QTreeWidgetItem([name, f"{value:.6g}", found.labels.get(name, "")])
            item.setData(0, Qt.ItemDataRole.UserRole, (lines.get(name, 0), 1))
            self.steady.addTopLevelItem(item)
        for row in found.roots_table().rows if found.roots else ():
            self.roots.addTopLevelItem(QtWidgets.QTreeWidgetItem([str(row[0]), f"{row[1]:.4f}", f"{row[2]:.4f}", f"{row[3]:.4f}", row[4]]))
        for tree in (self.steady, self.roots):
            for col in range(tree.columnCount()):
                tree.resizeColumnToContents(col)
        self.solution = found
        self.solved_text = self._checked()  # the text (and syntax) the Solution tab shows: stale once it changes
        self._set_irf(found)
        return found

    def _checked(self) -> tuple[str, Any]:
        return self.editor.toPlainText(), self.syntax.currentData()

    def _verdict(self, text: str, kind: str) -> None:
        self.verdict.setText(text)
        self.verdict.setStyleSheet(theme.css({"ok": "success", "error": "error"}.get(kind, "muted")))

    def source(self) -> Any:
        from ..models.source import ModelSource

        return ModelSource(self.editor.toPlainText(), self.syntax.currentData(), self.name.text().strip() or "model")

    def _syntax_changed(self) -> None:
        syntax = self.syntax.currentData()
        self.editor.highlighter.set_syntax(syntax)
        self.editor.syntax = syntax
        if not self.editor.toPlainText().strip() or self.editor.toPlainText() in TEMPLATES.values():
            self.editor.setPlainText(TEMPLATES[syntax])
        self.refresh()

    def refresh(self) -> None:
        """Check the text now: the problems, the symbols, the underlines and the status line."""
        self.timer.stop()  # a check still due from typing is this one
        found = self.source().check()
        self.last = found
        self.editor.found = found
        self.editor.set_diagnostics(list(found.diagnostics))
        self.problems.clear()
        for d in found.diagnostics:
            item = QtWidgets.QTreeWidgetItem([str(d.line), str(d.column), d.severity, d.message])
            item.setData(0, Qt.ItemDataRole.UserRole, (d.line, d.column))
            item.setToolTip(3, d.message)  # the whole message, when the column is narrow
            item.setForeground(2, theme.colour(d.severity if d.severity in ("error", "warning") else "muted"))
            self.problems.addTopLevelItem(item)
        if not found.diagnostics:  # an empty state across the row, as the main window's panels show theirs
            from .messages import EMPTY_STATES

            empty = QtWidgets.QTreeWidgetItem([EMPTY_STATES["model_problems"]])
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            empty.setToolTip(0, EMPTY_STATES["model_problems"])
            self.problems.addTopLevelItem(empty)
            empty.setFirstColumnSpanned(True)
        self.symbols.clear()
        for s in found.symbols:
            value = "" if s.value is None else f"{s.value:g}"
            item = QtWidgets.QTreeWidgetItem([s.name, s.role, value, s.description, ", ".join(map(str, s.used)), s.estimation])
            item.setData(0, Qt.ItemDataRole.UserRole, (s.line, 1))
            self.symbols.addTopLevelItem(item)
        for tree in (self.problems, self.symbols):
            for col in range(tree.columnCount() - 1):
                tree.resizeColumnToContents(col)
        errors = len(found.errors)
        warnings = sum(d.severity == "warning" for d in found.diagnostics)
        from .messages import plural

        self.status.setText(f" {plural(found.equations, 'equation')} for {plural(len(found.role('endogenous')), 'endogenous variable')}; "
                            f"{plural(errors, 'error')}, {plural(warnings, 'warning')}")
        self.sidecars.setTabText(0, f"Problems ({errors + warnings})" if errors + warnings else "Problems")
        self.apply_action.setEnabled(found.ok and found.equations > 0)
        self.solve_action.setEnabled(found.ok and found.equations > 0)
        if (self.steady.topLevelItemCount() or self.roots.topLevelItemCount()) and self._checked() != getattr(self, "solved_text", None):
            self._verdict("The text has changed since the last check: Check solution again.", "stale")
        self.equations_stale = True
        self._show_equations()

    def _go(self, item: QtWidgets.QTreeWidgetItem, _column: int = 0) -> None:
        line, column = item.data(0, Qt.ItemDataRole.UserRole) or (0, 1)
        block = self.editor.document().findBlockByNumber(max(line, 1) - 1)
        if not block.isValid():
            return
        cursor = QtGui.QTextCursor(block)
        cursor.setPosition(block.position() + min(max(column - 1, 0), max(block.length() - 1, 0)))
        self.editor.setTextCursor(cursor)
        self.editor.centerCursor()
        self.editor.setFocus()

    def export(self, syntax: str) -> Any:
        """The model written in ``syntax``, in a new editor (with its Translation notes); None, with the reason in the
        status line, when the text has errors or the model cannot be written there."""
        from ..errors import PyMetrixError

        try:
            translation = self.source().export(syntax)
        except (PyMetrixError, ValueError) as exc:
            self.status.setText(f" Could not write it as {LABELS[syntax]}: {exc}")
            return None
        name = self.name.text().strip() or "model"
        if self.open_other is not None:
            return self.open_other(translation.text, syntax, name, translation)
        other = ModelEditorWindow(self.apply_step, translation.text, syntax, name, translation=translation)
        other.setWindowTitle(f"{name} — Model editor ({LABELS[syntax]})")
        other.resize(self.size())
        other.show()
        self._others = [*getattr(self, "_others", []), other]  # kept alive
        return other

    def command(self) -> Any:
        from ..commands.model_source import DefineModel

        src = self.source()
        return DefineModel(src.text, src.syntax, src.name)

    def apply(self) -> bool:
        """Record the model as a step (``DefineModel``): False when the text has errors or the step failed."""
        self.refresh()
        if self.last is None or not self.last.ok or self.apply_step is None:
            return False
        return bool(self.apply_step(self.command()))
