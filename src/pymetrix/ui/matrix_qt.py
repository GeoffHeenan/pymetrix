"""Matrices in the app: Model > Matrix > Enter values (a grid to type or paste numbers into, with names), and a matrix
window's Matrix menu (Transpose, Inverse, Cholesky, Decompose, Power, Solve Lyapunov, Copy as LaTeX / CSV, Export).

Every action that makes a matrix runs a step (``pymetrix.commands.matrices``): it records ``name = <expression>``,
asks for the new matrix's name (a free one is offered), and opens the new matrix's window. The questions go through
the module's ``ask_*`` functions, which tests replace."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from PySide6 import QtGui, QtWidgets

from ..commands.matrices import (DECOMPOSE_ACTIONS, MATRIX_ACTIONS, EnterMatrix, action_step, format_number, lyapunov_step,
                                 parse_grid, parse_number, power_step, result_matrix_steps, unique_name)
from ..errors import SpecificationError
from ..matrix import Matrix
from ..matrix_expr import DECOMPOSITIONS
from . import theme

__all__ = ["MatrixGridDialog", "decomposition_menu", "matrix_menu", "result_matrix_menu"]


# -- questions (replaced in tests) ------------------------------------------------------------------------------------


def ask_name(parent: Any, title: str, default: str) -> str | None:
    text, ok = QtWidgets.QInputDialog.getText(parent, title, "Name of the new matrix:", text=default)
    return text.strip() if ok and text.strip() else None


def ask_int(parent: Any, title: str, label: str, default: int, low: int, high: int) -> int | None:
    value, ok = QtWidgets.QInputDialog.getInt(parent, title, label, default, low, high)
    return value if ok else None


def ask_item(parent: Any, title: str, label: str, items: list[str]) -> str | None:
    item, ok = QtWidgets.QInputDialog.getItem(parent, title, label, items, 0, False)
    return item if ok else None


def save_path(parent: Any, title: str, default: str) -> str:
    path, _ = QtWidgets.QFileDialog.getSaveFileName(parent, title, default, "Excel (*.xlsx)")
    return path


def warn(parent: Any, title: str, text: str) -> None:
    QtWidgets.QMessageBox.warning(parent, title, text)


def _copy(text: str) -> None:
    QtWidgets.QApplication.clipboard().setText(text)


# -- a matrix window's menu -------------------------------------------------------------------------------------------


def _named(parent: Any, cmd: Any, title: str, run: Callable[[Any], Any]) -> None:
    """Ask for the new matrix's name (the step's own offered), then run the step."""
    name = ask_name(parent, title, cmd.name)
    if name is not None:
        run(dataclasses.replace(cmd, name=name))


def export_excel(parent: Any, matrix: Matrix, name: str) -> str:
    """Write the matrix to an Excel file, its names as the first row and column (complex numbers as text)."""
    path = save_path(parent, "Export matrix to Excel", f"{name}.xlsx")
    if path:
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        frame = matrix.to_frame()
        (frame.astype(str) if matrix.is_complex else frame).to_excel(path, sheet_name=name[:31] or "matrix")
    return path


def matrix_menu(parent: Any, name: str, matrix: Matrix, session: Any, run: Callable[[Any], Any]) -> QtWidgets.QMenu:
    """The Matrix menu of a matrix's window: each step makes a new named matrix through ``run(step)``."""
    menu = QtWidgets.QMenu("Matrix", parent)
    taken = session.objects

    def step(label: str, expression: str, suffix: str) -> None:
        _named(parent, action_step(name, expression, suffix, taken), label, run)

    for label, expression, suffix in MATRIX_ACTIONS:
        menu.addAction(label).triggered.connect(lambda _=False, a=(label, expression, suffix): step(*a))
    decompose = menu.addMenu("Decompose")
    for label, expression, suffix in DECOMPOSE_ACTIONS:
        decompose.addAction(label).triggered.connect(lambda _=False, a=(label, expression, suffix): step(*a))
    menu.addSeparator()

    def power() -> None:
        n = ask_int(parent, "Matrix power", f"{name} to the power (whole number; negative: of the inverse):", 2, -50, 50)
        if n is not None:
            _named(parent, power_step(name, n, taken), "Matrix power", run)

    def lyapunov() -> None:
        size = matrix.shape[0]
        options = sorted(n for n, o in session.objects.items() if isinstance(o, Matrix) and o.shape == (size, size))
        if matrix.shape[0] != matrix.shape[1] or not options:
            warn(parent, "Solve Lyapunov", f"X = A X A' + Q needs a square A and a {size} x {size} matrix Q in the session"
                 " (make one with Model > Matrix)")
            return
        q = ask_item(parent, "Solve Lyapunov", f"X = {name} X {name}' + Q, with Q:", options)
        if q is not None:
            _named(parent, lyapunov_step(name, q, taken), "Solve Lyapunov", run)

    menu.addAction("Power…").triggered.connect(power)
    menu.addAction("Solve Lyapunov…").triggered.connect(lyapunov)
    menu.addSeparator()
    menu.addAction("Copy as LaTeX (bmatrix)").triggered.connect(lambda: _copy(matrix.to_latex()))
    menu.addAction("Copy as LaTeX table (with names)").triggered.connect(lambda: _copy(matrix.to_latex(names=True)))
    # lines end with \n everywhere: pandas would use the platform's own ending (\r\n on Windows); Excel takes either
    menu.addAction("Copy as CSV").triggered.connect(lambda: _copy(matrix.to_frame().to_csv(lineterminator="\n")))
    menu.addAction("Copy as tab-separated (for Excel)").triggered.connect(
        lambda: _copy(matrix.to_frame().to_csv(sep="\t", lineterminator="\n")))
    menu.addAction("Export to Excel…").triggered.connect(lambda: export_excel(parent, matrix, name))
    return menu


def decomposition_menu(parent: Any, name: str, factors: Any, session: Any, run: Callable[[Any], Any]) -> QtWidgets.QMenu:
    """A decomposition's window: each factor as a named matrix of its own (``U = A_svd.U``)."""
    menu = QtWidgets.QMenu("Matrix", parent)
    for field in factors._fields:
        menu.addAction(f"{field} as a matrix…").triggered.connect(
            lambda _=False, f=field: _named(parent, action_step(name, "{m}." + f, f, session.objects), f"{f} as a matrix", run))
    return menu


def result_matrix_menu(parent: Any, name: str, obj: Any, session: Any, run: Callable[[Any], Any]) -> QtWidgets.QMenu | None:
    """A fitted result's As matrix menu (coefficients, covariance; a VAR's lag matrices and error covariance)."""
    steps = result_matrix_steps(name, obj, session.objects)
    if not steps:
        return None
    menu = QtWidgets.QMenu("As matrix", parent)
    for label, cmd in steps:
        menu.addAction(label + "…").triggered.connect(lambda _=False, c=cmd, t=label: _named(parent, c, t, run))
    return menu


def menu_for(parent: Any, name: str, obj: Any, session: Any, run: Callable[[Any], Any]) -> QtWidgets.QMenu | None:
    """The Matrix menu for an object's window: a matrix's, a decomposition's, or None."""
    if isinstance(obj, Matrix):
        return matrix_menu(parent, name, obj, session, run)
    if isinstance(obj, DECOMPOSITIONS):
        return decomposition_menu(parent, name, obj, session, run)
    return None


# -- Model > Matrix > Enter values ------------------------------------------------------------------------------------


class MatrixGridDialog(QtWidgets.QDialog):
    """A grid to type or paste a matrix into: its size, its numbers, and names for its rows and columns (double-click
    a header to rename it). Paste (Ctrl+V) reads text from Excel or with tabs, semicolons, commas or spaces between
    the cells; a first row or column of names gives the names. OK runs the step (``EnterMatrix``)."""

    def __init__(self, session: Any, parent: Any = None, initial: Any = None) -> None:
        super().__init__(parent)
        self.session = session
        self.ran: EnterMatrix | None = None
        self.setWindowTitle("Enter matrix")
        layout = QtWidgets.QVBoxLayout(self)
        from .metrics import set_margins

        set_margins(layout, "dialog_margin")
        form = QtWidgets.QHBoxLayout()
        self.name = QtWidgets.QLineEdit(getattr(initial, "name", "") or unique_name("M", session.objects))
        self.rows = QtWidgets.QSpinBox()
        self.cols = QtWidgets.QSpinBox()
        for spin in (self.rows, self.cols):
            spin.setRange(1, 500)
            spin.setValue(2)
        for label, widget in (("Name", self.name), ("Rows", self.rows), ("Columns", self.cols)):
            form.addWidget(QtWidgets.QLabel(label))
            form.addWidget(widget)
        paste = QtWidgets.QPushButton("Paste")
        paste.setToolTip("numbers copied from Excel, or text with tabs, semicolons, commas or spaces between the cells; "
                         "a first row or column of names gives the names")
        paste.clicked.connect(lambda: self.paste_text(QtWidgets.QApplication.clipboard().text()))
        form.addWidget(paste)
        layout.addLayout(form)
        self.table = QtWidgets.QTableWidget(2, 2)
        self.table.horizontalHeader().sectionDoubleClicked.connect(lambda j: self._rename(j, across=True))
        self.table.verticalHeader().sectionDoubleClicked.connect(lambda i: self._rename(i, across=False))
        layout.addWidget(self.table, 1)
        hint = QtWidgets.QLabel("Type or paste the numbers; double-click a row or column header to name it.")
        hint.setStyleSheet(theme.muted_css())
        layout.addWidget(hint)
        self.error = QtWidgets.QLabel("")
        self.error.setStyleSheet(theme.css("error"))
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.row_names: list[str] = []
        self.col_names: list[str] = []
        self.set_size(2, 2)
        self.rows.valueChanged.connect(lambda n: self.set_size(n, self.cols.value()))
        self.cols.valueChanged.connect(lambda k: self.set_size(self.rows.value(), k))
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Paste, self.table,
                        activated=lambda: self.paste_text(QtWidgets.QApplication.clipboard().text()))
        if initial is not None and getattr(initial, "values", ""):
            try:
                values, rows, cols = initial.grid()
            except SpecificationError:
                pass
            else:
                self.fill(values, rows or None, cols or None)
        self.resize(560, 360)

    # the grid

    def set_size(self, n: int, k: int) -> None:
        """Rows and columns: new ones numbered, the names and numbers already there kept."""
        self.row_names = (self.row_names + [""] * n)[:n]
        self.col_names = (self.col_names + [""] * k)[:k]
        self.table.setRowCount(n)
        self.table.setColumnCount(k)
        for spin, value in ((self.rows, n), (self.cols, k)):
            if spin.value() != value:
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
        self._headers()

    def _headers(self) -> None:
        self.table.setVerticalHeaderLabels([r or str(i) for i, r in enumerate(self.row_names, 1)])
        self.table.setHorizontalHeaderLabels([c or str(j) for j, c in enumerate(self.col_names, 1)])

    def rename_row(self, i: int, text: str) -> None:
        self._set_name(self.row_names, i, text, "row")

    def rename_col(self, j: int, text: str) -> None:
        self._set_name(self.col_names, j, text, "column")

    def _set_name(self, names: list[str], at: int, text: str, what: str) -> None:
        text = text.strip()
        if "," in text:
            self.error.setText(f"a {what} name cannot hold a comma: {text!r}")
            return
        names[at] = text
        self._headers()

    def _rename(self, at: int, across: bool) -> None:
        names = self.col_names if across else self.row_names
        text, ok = QtWidgets.QInputDialog.getText(self, "Name", f"{'Column' if across else 'Row'} {at + 1}'s name (empty: its number):",
                                                  text=names[at])
        if ok:
            (self.rename_col if across else self.rename_row)(at, text)

    def fill(self, values: list[list[float]], rows: list[str] | None = None, cols: list[str] | None = None) -> None:
        n, k = len(values), len(values[0]) if values else 0
        self.row_names, self.col_names = list(rows or []), list(cols or [])
        self.set_size(n, k)
        for i, row in enumerate(values):
            for j, v in enumerate(row):
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(format_number(v)))

    def paste_text(self, text: str) -> bool:
        """Read pasted text into the grid, resizing it (and taking names from a first row or column of them)."""
        try:
            values, rows, cols = parse_grid(text)
        except SpecificationError as exc:
            self.error.setText(f"Could not paste: {exc}")
            return False
        if any("," in x for x in (rows or []) + (cols or [])):
            self.error.setText("Could not paste: a name holds a comma")
            return False
        self.fill(values, rows if rows is not None else self.row_names if len(self.row_names) == len(values) else None,
                  cols if cols is not None else self.col_names if len(self.col_names) == len(values[0]) else None)
        self.error.setText("")
        return True

    def command(self) -> EnterMatrix:
        """The step the grid describes (raises SpecificationError for an empty or non-numeric cell)."""
        lines = []
        for i in range(self.table.rowCount()):
            cells = []
            for j in range(self.table.columnCount()):
                item = self.table.item(i, j)
                text = item.text().strip() if item is not None else ""
                if not text:
                    raise SpecificationError(f"row {i + 1}, column {j + 1} is empty")
                try:
                    cells.append(format_number(parse_number(text)).replace("np.", ""))
                except ValueError:
                    raise SpecificationError(f"row {i + 1}, column {j + 1}: {text!r} is not a number") from None
            lines.append("\t".join(cells))
        named = lambda names: ", ".join(x or str(k) for k, x in enumerate(names, 1)) if any(names) else ""  # noqa: E731
        return EnterMatrix("\n".join(lines), rows=named(self.row_names), cols=named(self.col_names), name=self.name.text().strip())

    def accept(self) -> None:
        try:
            cmd = self.command()
        except SpecificationError as exc:
            self.error.setText(str(exc))
            return
        errors = cmd.validate(self.session)
        if errors:
            self.error.setText("; ".join(errors))
            return
        try:
            self.session.run(cmd)
        except Exception as exc:  # noqa: BLE001 - shown in the dialog, which stays open
            self.error.setText(str(exc))
            return
        self.ran = cmd
        super().accept()
