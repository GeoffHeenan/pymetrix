"""Named matrices in the app: Model > Matrix (an expression, from data or a result, values in a grid), a matrix window's
Matrix menu, a fitted result's As matrix, and the help topic.

Checked here:

- every matrix step gives exactly the matrix its recorded code gives: the session's script, run in a fresh namespace,
  makes the same matrices (names and values), decompositions included;
- each new command round-trips through a project (``check_project``: saved, reopened, the same result and script),
  with its inputs made by steps;
- a step is recomputed when the objects its expression uses change, and reads only those;
- expressions are checked before they run: unknown names (with the nearest), syntax errors, statements, imports,
  dunder attributes, lambdas, comprehensions, exec / eval and file input or output are refused with clear messages;
- the grid's paste parsing (tabs from Excel, semicolons and spaces with decimal commas, commas, a first row or column
  of names, ragged or empty cells), its sizes and name editing, and the literal it records;
- LaTeX (a plain bmatrix, or a table with the names) and the copy and export actions;
- in the app (offscreen): the Matrix menu of a matrix's and a decomposition's window, Power and Solve Lyapunov, the
  objects panel's As matrix, the Next menu's As a matrix, Model > Matrix and the grid dialog.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from pymetrix import Matrix
from pymetrix.commands import COMMANDS, DefineMatrix, EnterMatrix, FitEquation, FitVAR, LoadExample, MatrixFrom, Session
from pymetrix.commands.matrices import matrix_literal, parse_grid, parse_number, result_matrix_steps
from pymetrix.errors import SpecificationError
from pymetrix.matrix_expr import DECOMPOSITIONS, ExpressionError, check_expression, evaluate
from pymetrix.testing import check_project


@pytest.fixture
def session():
    s = Session()
    s.run(LoadExample("sample_macro"))
    s.run(FitEquation("dlog(c) ~ dlog(y) + dlog(c)[-1]", name="fit"))
    s.run(FitVAR("c, y", lags="2", name="var"))
    return s


def _same(a, b):
    if isinstance(a, DECOMPOSITIONS):
        assert type(a) is type(b)
        for x, y in zip(a, b):
            _same(x, y)
        return
    assert isinstance(a, Matrix) and isinstance(b, Matrix)
    assert a.rows == b.rows and a.cols == b.cols
    np.testing.assert_allclose(a.values, b.values, rtol=1e-12, atol=1e-14)


STEPS = [
    MatrixFrom("coefficients", result="fit", name="b"),
    MatrixFrom("covariance", result="fit", name="V"),
    MatrixFrom("lag matrix", var="var", lag=2, name="A2"),
    MatrixFrom("error covariance", var="var", name="S"),
    MatrixFrom("series", data="data", series="c, y", start="2001Q1", end="2004Q4", name="Y"),
    MatrixFrom("series", data="data", name="D"),
    MatrixFrom("design", formula="c ~ y + c[-1]", data="data", start="2001Q1", name="X", dependent="yy"),
    DefineMatrix("(X.T @ X).inv() @ X.T @ yy", name="beta"),
    DefineMatrix("V.chol(upper=True)", name="R"),
    DefineMatrix("S.svd()", name="S_svd"),
    DefineMatrix("S_svd.U", name="U"),
    DefineMatrix("Matrix.lyapunov(A2, S)", name="L"),
    DefineMatrix("A2.kron(Matrix.identity(2)) * 2 - 1", name="K"),
    DefineMatrix("np.log(abs(S)) + Matrix.identity(['c', 'y'])", name="logS"),
    EnterMatrix("1\t0.5\n0.5\t2", rows="a, b", cols="a, b", name="G"),
    EnterMatrix("1 -2.25 3e-3", name="g"),
    DefineMatrix("G.power(-2)", name="G2"),
]


def test_each_step_gives_the_matrix_its_recorded_code_gives(session):
    for step in STEPS:
        session.run(step)
    script = session.script()
    assert "beta = (X.T @ X).inv() @ X.T @ yy" in script and "X, yy = Matrix.design('c ~ y + c[-1]', data, span=('2001Q1', None))" in script
    assert "G = Matrix([[1, 0.5], [0.5, 2]], rows=['a', 'b'], cols=['a', 'b'])" in script and "g = Matrix([[1, -2.25, 0.003]])" in script
    assert "A2 = Matrix.var_lag(var, 2)" in script and "Y = Matrix.from_dataset(data, ['c', 'y'], span=('2001Q1', '2004Q4'))" in script
    fresh: dict = {}
    exec(compile(script, "<script>", "exec"), fresh)  # noqa: S102 - the session's own script
    names = {"b", "V", "A2", "S", "Y", "D", "X", "yy", "beta", "R", "S_svd", "U", "L", "K", "logS", "G", "g", "G2"}
    for name in names:
        _same(session[name], fresh[name])
    for record in session.history[-len(STEPS):]:  # the recorded call (the design's two matrices have none) writes the same
        assert record.ok and (record.call is None or record.call.to_code() == record.code)
    assert sum(r.call is None for r in session.history[-len(STEPS):]) == 1


def test_the_steps_read_what_their_expression_uses_and_follow_it(session):
    session.run(MatrixFrom("covariance", result="fit", name="V"))
    session.run(DefineMatrix("V.chol() @ V.chol().T - V", name="Z"))
    assert set(session.history[-1].reads) == {"V"} and set(session.history[-2].reads) == {"fit"}
    before = session["V"].shape
    session.run(FitEquation("dlog(c) ~ dlog(y)", name="fit"))  # fewer coefficients: V and Z follow
    assert set(session.stale()) == {"V", "Z"}
    session.recompute()
    assert session["V"].shape == (2, 2) != before and session["Z"].shape == (2, 2)
    np.testing.assert_allclose(session["Z"].values, 0, atol=1e-15)


@pytest.mark.parametrize("step", STEPS[:7] + [STEPS[8], STEPS[9], STEPS[14]], ids=lambda s: f"{type(s).__name__}-{s.name}")
def test_each_command_round_trips_through_a_project(session, step, tmp_path):
    for earlier in STEPS[: STEPS.index(step)]:
        if earlier.name not in session.objects:
            session.run(earlier)
    check_project(session, step, tmp_path)


@pytest.mark.parametrize("text, message", [
    ("", "write a matrix expression"),
    ("Vx.T", "no object named 'Vx' (did you mean V?)"),
    ("V +", "not a Python expression"),
    ("V = 1", "without 'name ='"),
    ("import os", "imports are not allowed"),
    ("V; V", "one expression only"),
    ("V.__class__", "attributes starting with an underscore are not allowed (__class__)"),
    ("V._meet", "attributes starting with an underscore"),
    ("__import__('os')", "__import__ is not allowed"),
    ("lambda: V", "lambda is not allowed"),
    ("[x for x in V]", "a comprehension is not allowed"),
    ("exec('1')", "exec is not allowed"),
    ("eval('V')", "eval is not allowed"),
    ("open('f')", "open is not allowed"),
    ("getattr(V, 'T')", "getattr is not allowed"),
    ("np.save('f', V.values)", "save reads or writes files"),
    ("V.to_frame().to_csv('f')", "to_csv reads or writes files"),
    ("V if V else V", "if ... else is not allowed"),
    ("Matrix.cov(**{'result': V})", "** unpacking is not allowed"),
])
def test_expressions_are_checked_before_they_run(text, message):
    with pytest.raises(ExpressionError, match=None) as caught:
        check_expression(text, {"V": None})
    assert message in str(caught.value)
    assert message in "; ".join(DefineMatrix(text, name="M").validate(_with_v()))


def _with_v():
    s = Session()
    s.objects["V"] = Matrix.identity(2)
    return s


def test_an_expression_must_give_a_matrix_and_names_must_be_usable():
    s = _with_v()
    with pytest.raises(ExpressionError, match="gives a number, not a matrix"):
        evaluate("V.det()", s.objects)
    with pytest.raises(ExpressionError, match="NumPy array"):
        evaluate("np.eye(2)", s.objects)
    assert isinstance(evaluate("V.eig()", s.objects), DECOMPOSITIONS)
    assert DefineMatrix("V.T", name="2x").validate(s) == ["Name: '2x' is not a Python name (letters, digits and _, not starting with a digit)"]
    assert "is taken" in DefineMatrix("V.T", name="np").validate(s)[0]
    assert MatrixFrom("covariance", result="nope").validate(s) == ["no object named 'nope'"]
    assert MatrixFrom("series", data="V", series="q").validate(s) == ["V has no series q"]
    assert "different names" in "; ".join(MatrixFrom("design", formula="c ~ y", name="y", dependent="y").validate(s))
    with pytest.raises(SpecificationError, match="not positive definite"):
        s.run(DefineMatrix("(-V).chol()", name="W"))
    assert not s.history[-1].ok and "W" not in s.objects


def test_the_grid_reads_pasted_text():
    assert parse_grid("1\t2\n3\t4\n") == ([[1, 2], [3, 4]], None, None)  # Excel's copy
    assert parse_grid("1;2,5\n-3;4") == ([[1, 2.5], [-3, 4]], None, None)  # semicolons and a decimal comma
    assert parse_grid("1 2,5\n3 \u22124") == ([[1, 2.5], [3, -4]], None, None)  # spaces, a decimal comma, a Unicode minus
    assert parse_grid("1, 2.5\n3, 4e-2") == ([[1, 2.5], [3, 0.04]], None, None)  # commas between the cells
    assert parse_grid("\tx\ty\na\t1\t2\nb\t3\t4") == ([[1, 2], [3, 4]], ["a", "b"], ["x", "y"])  # names both ways
    assert parse_grid("x\ty\n1\t2") == ([[1, 2]], None, ["x", "y"])
    assert parse_grid("a\t1\t2\nb\t3\t4") == ([[1, 2], [3, 4]], ["a", "b"], None)
    assert parse_grid("1\t2\t\t\n3\t4\t\t") == ([[1, 2], [3, 4]], None, None)  # a spreadsheet's empty columns
    for text, message in (("", "nothing to read"), ("1\t2\n3", "row 2, column 2 is empty"), ("1\t2\n3\tx", "'x' is not a number"),
                          ("1 2\n3 4 5", "row 2 has 3 cells")):
        with pytest.raises(SpecificationError, match=message):
            parse_grid(text)
    assert parse_number(" 1,5 ") == 1.5 and np.isnan(parse_number("nan"))
    with pytest.raises(ValueError):
        parse_number("1.2.3")
    assert matrix_literal([[1.0, 0.25]], ["1"], ["a", "b"]) == "Matrix([[1, 0.25]], cols=['a', 'b'])"


def test_values_entered_need_names_that_fit():
    s = Session()
    assert EnterMatrix("1 2\n3 4", rows="a").validate(s) == ["1 row names for 2 rows"]
    assert EnterMatrix("1 2\n3 4", cols="a, a").validate(s) == ["column names must differ: 'a' is used twice"]
    assert EnterMatrix("x\ty\n1\t2").validate(s) == ["the values are numbers only: give the names under Rows and Columns"]
    s.run(EnterMatrix("1 2\n3 4", rows="r1, r2", cols="c1, c2", name="M"))
    assert s["M"].rows == ("r1", "r2") and s["M"]["r2", "c1"] == 3.0


def test_latex():
    m = Matrix([[1, -0.5], [0.25, 2]], rows=["a_1", "b"], cols=["x", "y%"])
    assert m.to_latex() == "\\begin{bmatrix}\n  1 & -0.5 \\\\\n  0.25 & 2\n\\end{bmatrix}"
    table = m.to_latex(names=True)
    assert table.startswith("\\begin{tabular}{lrr}\n\\hline\n & x & y\\% \\\\\n\\hline\n") and "a\\_1 & 1 & -0.5" in table
    assert Matrix(np.array([[1 + 2j]])).to_latex() == "\\begin{bmatrix}\n  1+2i\n\\end{bmatrix}"


def test_results_offer_their_matrices(session):
    labels = [label for label, _ in result_matrix_steps("var", session["var"], session.objects)]
    assert labels == ["Coefficients", "Covariance of the estimates", "Lag 1 matrix", "Lag 2 matrix", "Error covariance"]
    assert [label for label, _ in result_matrix_steps("fit", session["fit"], session.objects)] == ["Coefficients", "Covariance of the estimates"]
    assert result_matrix_steps("data", session["data"], session.objects) == []
    steps = dict(result_matrix_steps("var", session["var"], session.objects))
    session.run(steps["Lag 2 matrix"])
    assert session.history[-1].code == "var_A2 = Matrix.var_lag(var, 2)"


def test_the_help_topic_and_the_menus():
    from pymetrix.commands.menus import menu_order
    from pymetrix.help import topic

    t = topic("matrices")
    assert "Model > Matrix" in t.text and any("guide.md#matrices-with-names" in r for r in t.references)
    assert DefineMatrix.topics == MatrixFrom.topics == EnterMatrix.topics == ("matrices",)
    paths = dict((i.cls.__name__, p) for i, p in menu_order([i for i in COMMANDS.values() if i.menu]))
    assert paths["DefineMatrix"] == "Model → Matrix → Expression…" and paths["MatrixFrom"] == "Model → Matrix → From data or a result…"
    assert paths["EnterMatrix"] == "Model → Matrix → Enter values…"


# -- the app ----------------------------------------------------------------------------------------------------------


@pytest.fixture
def app_window(session):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import main_window

    session.run(MatrixFrom("error covariance", var="var", name="S"))
    win = main_window(session)
    yield win, session
    session.dirty = False
    win.close()


def _find(menu, text):
    """The action with this text in a menu or its submenus. Found through the menu's children, not QAction.menu():
    under PySide6 6.6 the wrapper that menu() returns can take the submenu with it when it is collected."""
    from PySide6.QtGui import QAction

    return next((a for a in menu.findChildren(QAction) if a.text().replace("&", "") == text), None)


def _action(menu, text):
    found = _find(menu, text)
    assert found is not None, f"no action {text!r} in {[a.text().replace('&', '') for a in menu.actions()]}"
    return found


def test_copied_text_ends_lines_with_newline_on_every_platform(app_window, monkeypatch):
    """CSV and tab-separated copies end lines with \\n even where pandas would use the platform's line ending
    (\\r\\n on Windows, simulated here), so the clipboard holds the same text everywhere."""
    import os

    from PySide6.QtWidgets import QApplication

    win, s = app_window
    monkeypatch.setattr(os, "linesep", "\r\n")
    win.show_object("S")
    menu = win.mdi.activeSubWindow().widget().matrix_menu
    for text, sep in (("Copy as CSV", ","), ("Copy as tab-separated (for Excel)", "\t")):
        _action(menu, text).trigger()
        copied = QApplication.clipboard().text()
        assert "\r" not in copied and copied == s["S"].to_frame().to_csv(sep=sep, lineterminator="\n"), text


def test_a_matrix_window_makes_new_matrices_as_steps(app_window, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QApplication

    from pymetrix.ui import matrix_qt

    win, s = app_window
    names = iter(["St", "Sinv", "Rup", "Ssvd", "S3", "Slyap"])
    monkeypatch.setattr(matrix_qt, "ask_name", lambda parent, title, default: next(names))
    monkeypatch.setattr(matrix_qt, "ask_int", lambda *a: 3)
    monkeypatch.setattr(matrix_qt, "ask_item", lambda parent, title, label, items: items[items.index("S")])
    win.show_object("S")
    menu = win.mdi.activeSubWindow().widget().matrix_menu
    labels = [a.text().replace("&", "") for a in menu.actions() if a.text()]
    assert labels == ["Transpose", "Inverse", "Pseudo-inverse", "Cholesky factor (lower)", "Cholesky factor (upper)", "Decompose",
                      "Power…", "Solve Lyapunov…", "Copy as LaTeX (bmatrix)", "Copy as LaTeX table (with names)", "Copy as CSV",
                      "Copy as tab-separated (for Excel)", "Export to Excel…"]
    for text in ("Transpose", "Inverse", "Cholesky factor (upper)", "Singular values (SVD)", "Power…"):
        _action(menu, text).trigger()
    s.run(DefineMatrix("S.T @ Matrix.identity(['c', 'y']) * 0", name="F"))  # A for the Lyapunov equation: F X F' + S
    win.show_object("F")
    _action(win.mdi.activeSubWindow().widget().matrix_menu, "Solve Lyapunov…").trigger()
    codes = [r.code for r in s.history[-7:]]
    assert codes == ["St = S.T", "Sinv = S.inv()", "Rup = S.chol(upper=True)", "Ssvd = S.svd()", "S3 = S.power(3)",
                     "F = S.T @ Matrix.identity(['c', 'y']) * 0", "Slyap = Matrix.lyapunov(F, S)"]
    _same(s["Slyap"], s["S"])  # F is zero: X = Q
    _same(s["S3"], s["S"] @ s["S"] @ s["S"])
    assert win.mdi.activeSubWindow().windowTitle().startswith("Slyap")
    win.show_object("Ssvd")  # a decomposition: each factor as a matrix
    factors = win.mdi.activeSubWindow().widget().matrix_menu
    assert [a.text().replace("&", "") for a in factors.actions()] == ["U as a matrix…", "s as a matrix…", "V as a matrix…"]
    monkeypatch.setattr(matrix_qt, "ask_name", lambda parent, title, default: default)
    factors.actions()[1].trigger()
    assert s.history[-1].code == "Ssvd_s = Ssvd.s" and s["Ssvd_s"].cols == ("singular value",)
    win.show_object("S")
    menu = win.mdi.activeSubWindow().widget().matrix_menu
    _action(menu, "Copy as LaTeX table (with names)").trigger()
    assert QApplication.clipboard().text() == s["S"].to_latex(names=True)
    _action(menu, "Copy as tab-separated (for Excel)").trigger()
    assert QApplication.clipboard().text().startswith("\tc\ty\nc\t")
    pytest.importorskip("openpyxl")
    import pandas as pd

    monkeypatch.setattr(matrix_qt, "save_path", lambda *a: str(tmp_path / "S"))
    _action(menu, "Export to Excel…").trigger()
    back = pd.read_excel(tmp_path / "S.xlsx", index_col=0)
    np.testing.assert_allclose(back.to_numpy(), s["S"].values)


def test_a_failed_action_says_why_and_records_nothing_new(app_window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from pymetrix.ui import matrix_qt

    win, s = app_window
    said = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda parent, title, text: said.append(text)))
    monkeypatch.setattr(matrix_qt, "ask_name", lambda parent, title, default: default)
    s.run(DefineMatrix("S - S", name="Zero"))
    win.show_object("Zero")
    _action(win.mdi.activeSubWindow().widget().matrix_menu, "Inverse").trigger()
    assert said and "singular" in said[-1] and "Zero_inv" not in s.objects


def test_fitted_results_offer_as_matrix(app_window, monkeypatch):
    from pymetrix.ui import matrix_qt

    win, s = app_window
    monkeypatch.setattr(matrix_qt, "ask_name", lambda parent, title, default: default)
    from PySide6.QtCore import Qt

    item = win.objects.findItems("var", Qt.MatchFlag.MatchExactly)[0]
    win.objects_menu(win.objects.visualItemRect(item).center())
    assert "As matrix" in win.last_objects_menu
    _action(win.last_objects_qmenu, "Lag 1 matrix…").trigger()
    assert s.history[-1].code == "var_A1 = Matrix.var_lag(var, 1)" and "var_A1" in s.objects
    win.show_object("fit")
    assert "As a matrix…" in [a.text().replace("&", "") for a in win.mdi.activeSubWindow().widget().next_menu.actions()]


def test_model_matrix_menu_and_the_grid(app_window, monkeypatch):
    from pymetrix.ui import matrix_qt

    win, s = app_window
    model = win.command_menus["Model"]
    matrix = next(a.menu() for a in model.actions() if a.menu() and a.menu().title().replace("&", "") == "Matrix")
    assert [a.text().replace("&", "") for a in matrix.actions()] == ["Expression…", "From data or a result…", "Enter values…"]
    opened = []
    monkeypatch.setattr(matrix_qt.MatrixGridDialog, "exec", lambda self: opened.append(self) or False)
    matrix.actions()[2].trigger()
    grid = opened[-1]
    assert grid.name.text() == "M" and (grid.table.rowCount(), grid.table.columnCount()) == (2, 2)
    assert grid.paste_text("\tx\ty\tz\na\t1\t2,5\t3\nb\t4\t5\t6\n")  # from Excel: names and a decimal comma
    assert (grid.rows.value(), grid.cols.value()) == (2, 3) and grid.col_names == ["x", "y", "z"]
    grid.rename_row(1, "beta")
    grid.rename_col(0, "a,b")  # refused: names are recorded separated by commas
    assert grid.col_names[0] == "x" and "comma" in grid.error.text()
    grid.cols.setValue(2)  # smaller: the last column goes
    grid.name.setText("G")
    cmd = grid.command()
    assert cmd == EnterMatrix("1\t2.5\n4\t5", rows="a, beta", cols="x, y", name="G")
    grid.accept()
    assert grid.ran == cmd and s.history[-1].code == "G = Matrix([[1, 2.5], [4, 5]], rows=['a', 'beta'], cols=['x', 'y'])"
    grid.table.item(0, 0).setText("")
    with pytest.raises(SpecificationError, match="row 1, column 1 is empty"):
        grid.command()
    assert not grid.paste_text("1\t2\n3") and "Could not paste" in grid.error.text()
    again = matrix_qt.MatrixGridDialog(s, win, cmd)  # reopening the step: its numbers and names
    assert again.command() == cmd
