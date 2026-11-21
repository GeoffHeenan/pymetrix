"""Models written as text (pymetrix.models.source) and the model editor: problems at their line and column, the
symbols, the model built, and the text kept as written by the step that builds it.

Checked: the New Keynesian model written in pymetrix's syntax builds the same solution as the template written in
Python, and Dynare's text the same model as read_mod; each planted mistake is reported once, where it is (no cascade
from one mistake to the lines after it); a project keeps the text, comments and all, and builds the model from it."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from pymetrix import ModelSource, read_iris, read_mod, templates
from pymetrix.models import ModelError

sys.path.insert(0, str(Path(__file__).parent))
from test_dynare import NK as NK_MOD  # noqa: E402

NK = '''# a New Keynesian model
parameters
    beta = 0.99        "discount factor"
    sigma = 1
    kappa = 0.1
    phi_pi = 1.5
    phi_y = 0.5
    rho = 0.8
shocks
    ey = 0.5   "demand"
    epi = 0.3
    ei = 0.2
equations
    IS: y = y[+1] - (1/sigma)*(i - pi[+1]) + ey     "the IS curve"
    Phillips: pi = beta*pi[+1] + kappa*y + epi
    policy: i = rho*i[-1] + (1 - rho)*(phi_pi*pi
                + phi_y*y) + ei
measurement
    infl = 4*pi  error 0.1  "inflation, annual"
'''


def _problems(text, syntax="pymetrix"):
    return [(d.line, d.column, d.severity, d.message) for d in ModelSource(text, syntax).check().diagnostics]


def test_pymetrix_syntax_builds_the_templates_model():
    found = ModelSource(NK).check()
    assert found.ok and not found.diagnostics and found.equations == 3
    assert found.role("endogenous") == ("y", "pi", "i") and found.role("shock") == ("ey", "epi", "ei")
    assert (found.max_lead, found.max_lag) == (1, 1)
    beta = next(s for s in found.symbols if s.name == "beta")
    assert (beta.role, beta.value, beta.description, beta.used) == ("parameter", 0.99, "discount factor", (15,))
    nk = ModelSource(NK, name="nk").build()
    ref = templates.new_keynesian()
    assert nk.shocks == ("ey", "epi", "ei") and nk.shock_sd == {"ey": 0.5, "epi": 0.3, "ei": 0.2}
    assert nk.labels["y"] == "the IS curve" and nk.measurement.errors == {"infl": 0.1}
    a, b = nk.solve(), ref.solve()
    order = [a.variables.index(v) for v in b.variables]
    np.testing.assert_allclose(a.P[np.ix_(order, order)], b.P, atol=1e-12)
    np.testing.assert_allclose(a.Q[order], b.Q, atol=1e-12)


def test_each_mistake_once_where_it_is():
    bad = (NK.replace("kappa*y", "kapa*y").replace("(1/sigma)*(i - pi[+1])", "(1/sigma)*(i - pi[+1]")
           .replace("phi_y = 0.5", "phi_y = 0.5 +"))
    errors = [p for p in _problems(bad) if p[2] == "error"]
    assert [(line, col) for line, col, _, _ in errors] == [(7, 18), (14, 5), (15, 34)]
    assert "never closed" in errors[1][3] and "did you mean 'kappa'" in errors[2][3]
    assert ModelSource(bad).check().equations == 2  # the unclosed one is not counted, and nothing after it is lost


@pytest.mark.parametrize("change, where, message", [
    (("    Phillips: pi =", "    Phillips: y ="), (15, 15), "already determined by the equation on line 14"),
    (("    Phillips: pi =", "    Phillips: pi[-1] ="), (15, 15), "the left-hand side holds the variable"),
    (("    Phillips: pi =", "    Phillips: pi + y ="), (15, 15), "and no other"),
    (("    beta = 0.99", "    beta = gamma + 1"), (3, 12), "gamma has no value yet"),
    (("infl = 4*pi", "infl = 4*pii"), (19, 12), "not a model variable or parameter; did you mean 'pi'"),
    (("# a New Keynesian model", "a New Keynesian model"), (1, 1), "text before any section"),
    (("    ei = 0.2", "    ei = 0.2\n    eu = 0.1"), (13, 5), "shock eu is not used"),
    (("    sigma = 1", "    sigma = 1\n    sigma = 2"), (5, 5), "sigma is declared twice"),
])
def test_problems_and_their_places(change, where, message):
    found = [p for p in _problems(NK.replace(*change)) if message in p[3]]
    assert found and found[0][:2] == where, _problems(NK.replace(*change))


def test_dynare_syntax():
    found = ModelSource(NK_MOD, "dynare").check()
    assert found.ok and found.equations == 4 and found.role("endogenous") == ("y", "pi", "i", "r")
    assert next(s for s in found.symbols if s.name == "r").description == "real rate"  # long_name
    assert next(s for s in found.symbols if s.name == "phi").value == 1.5
    built, direct = ModelSource(NK_MOD, "dynare").build(), read_mod(NK_MOD)
    np.testing.assert_allclose(built.solve().P, direct.solve().P, atol=1e-12)
    bad = NK_MOD.replace("kap*y", "kapp*y").replace("+ ei;", "+ eii;")
    errors = [p for p in _problems(bad, "dynare") if p[2] == "error"]
    assert [(line, col) for line, col, _, _ in errors] == [(10, 37), (11, 44)]
    assert "did you mean 'kap'" in errors[0][3] and "did you mean 'ei'" in errors[1][3]
    assert [p[:2] for p in _problems(bad, "dynare") if "parameter kap is not used" in p[3]] == [(5, 20)]
    unclosed = _problems(NK_MOD.replace("r = i - pi(+1);", "r = i - (pi(+1);"), "dynare")
    assert unclosed[0][:3] == (12, 1, "error") and "cannot read this equation" in unclosed[0][3]
    assert "unknown macro directive" in _problems("@#bogus N = 2\n" + NK_MOD, "dynare")[0][3]  # macros: test_dynare_macros.py
    assert not _problems("@#define N = 2\n" + NK_MOD, "dynare")  # a directive the macro processor carries out
    with pytest.raises(ModelError, match="the model text has errors"):
        ModelSource(bad, "dynare").build()


def test_the_step_keeps_the_text(tmp_path):
    from pymetrix.commands import Session, load_project, save_project
    from pymetrix.commands.model_source import DefineModel

    s = Session()
    s.run(DefineModel(NK, name="nk"))
    s.run(DefineModel(NK_MOD, syntax="dynare", name="dyn"))
    assert type(s["nk"]).__name__ == "Template" and type(s["dyn"]).__name__ == "DynareModel"
    assert 'nk = ModelSource("""\n# a New Keynesian model' in s.script() and "syntax='dynare'" in s.script()
    ns = {}
    exec(s.script(), ns)
    np.testing.assert_allclose(ns["nk"].solve().P, s["nk"].solve().P, atol=1e-14)
    reopened = load_project(save_project(s, tmp_path / "models.pymx"))
    kept = [r.command for r in reopened.history if type(r.command).__name__ == "DefineModel"]
    assert [c.text for c in kept] == [NK, NK_MOD]  # as written: comments, layout, descriptions
    np.testing.assert_allclose(reopened["dyn"].solve().P, s["dyn"].solve().P, atol=1e-14)
    assert DefineModel("parameters\n  a = \n", name="m").validate(s)  # errors stop the step


def test_the_editor_window():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.model_editor_qt import ModelEditorWindow

    applied = []
    window = ModelEditorWindow(lambda c: applied.append(c) or True, text=NK.replace("kappa*y", "kapa*y"), name="nk")
    assert not window.apply_action.isEnabled() and window.problems.topLevelItemCount() >= 1
    items = [window.problems.topLevelItem(i) for i in range(window.problems.topLevelItemCount())]
    first = next(item for item in items if "kapa" in item.text(3))
    assert (first.text(0), first.text(1), first.text(2)) == ("15", "34", "error")
    window._go(first)
    cursor = window.editor.textCursor()
    assert (cursor.blockNumber() + 1, cursor.positionInBlock() + 1) == (15, 34)
    assert any(sel.cursor.selectedText() == "kapa" for sel in window.editor.extraSelections())  # underlined where it is
    window.editor.setPlainText(NK)
    window.refresh()
    assert window.apply_action.isEnabled() and window.symbols.topLevelItemCount() == 3 + 3 + 6 + 1
    assert window.apply() and applied[0].text == NK and applied[0].name == "nk"
    window.syntax.setCurrentIndex(window.syntax.findData("dynare"))
    assert window.editor.highlighter.syntax == "dynare"


def test_from_the_main_window():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.commands import Session
    from pymetrix.ui.qt import main_window

    session = Session()
    window = main_window(session)
    try:
        labels = [a.text().replace("&", "") for a in window.command_menus["Model"].actions()]
        assert "Model editor…" in labels or "Model editor..." in labels
        editor = window.open_model_editor()
        editor.editor.setPlainText(NK)
        editor.name.setText("nk")
        assert editor.apply() and type(session["nk"]).__name__ == "Template"
        again = window.open_model_editor(session.history[-1].command)  # a step reopened: its own text
        assert again.editor.toPlainText() == NK and again.name.text() == "nk"
    finally:
        session.dirty = False  # closing does not ask to save
        window.close()


# --- phase 2: IRIS's syntax, and writing models in each syntax -------------------------------------

from test_iris import IRIS_NK, IRIS_RBC  # noqa: E402

IRIS_OK = IRIS_NK.replace("rho = 0.8, phi", "rho = 0.8, phi = 1.5")


def test_iris_syntax():
    from pymetrix import read_iris

    found = ModelSource(IRIS_OK, "iris").check()
    assert found.ok and found.equations == 4 and (found.max_lead, found.max_lag) == (1, 1)
    assert next(s for s in found.symbols if s.name == "y").description == "Output gap"
    built = ModelSource(IRIS_OK, "iris").build()
    np.testing.assert_allclose(built.solve().P, read_iris(IRIS_NK, params={"phi": 1.5}).solve().P, atol=1e-14)
    missing = _problems(IRIS_NK, "iris")  # phi has no value: at its declaration
    assert missing[0][:2] == (10, 48) and "no values for the parameters ['phi']" in missing[0][3]
    assert [p[:2] for p in _problems(IRIS_OK.replace("kap*y", "kapp*y"), "iris") if p[2] == "error"] == [(18, 34)]
    section = _problems(IRIS_OK.replace("!transition_equations", "!transition_equatons"), "iris")
    assert any("did you mean '!transition_equations'" in p[3] for p in section)


def test_iris_toolkit_dialect_in_the_editor():
    """The editor reads the QPM toolkit's dialect as read_iris does (one section table and splitter, shared): the
    fixture that read_iris reads checks clean (no "unknown section !modelname"), builds to the same model, keeps the
    descriptions after names, ignores // and # comments; and real errors in that dialect are placed at their line."""
    from test_iris import QPM_TOOLKIT_NK as qpm

    found = ModelSource(qpm, "iris").check()
    assert found.ok and not found.diagnostics and found.equations == 4
    described = {s.name: s.description for s in found.symbols}
    assert described["y"] == "Output gap (% of potential)" and described["pi"] == "Inflation" and described["i"] == ""
    assert {s.name for s in found.symbols} >= {"y", "pi", "i", "r", "ey", "me_pi", "obs_y", "phi", "std_me_pi"}
    assert not {"IS", "Phillips", "is_curve", "real_rate"} & {s.name for s in found.symbols}
    built, read = ModelSource(qpm, "iris").build(), read_iris(qpm)
    assert built.model.endogenous == read.model.endogenous and built.shock_sd == read.shock_sd
    assert built.labels == read.labels and built.description == read.description
    a, b = built.system(), read.system()
    for m in ("T", "R", "Q", "Z", "d", "H"):
        np.testing.assert_array_equal(getattr(a, m), getattr(b, m), err_msg=m)
    lines = qpm.split("\n")

    def place(text, line_text, word):  # (line, column) of word in the line that starts with line_text, from 1
        n = next(k for k, t in enumerate(text.split("\n")) if t.startswith(line_text))
        return n + 1, text.split("\n")[n].index(word) + 1

    typo = qpm.replace("kap*y + epi", "kapp*y + epi")  # in a named equation
    errors = [p for p in _problems(typo, "iris") if p[2] == "error"]
    assert [p[:2] for p in errors] == [place(typo, "phillips:", "kapp")] == [(25, 29)] and "kapp is not declared" in errors[0][3]
    bogus = qpm.replace("!assignments", "!Bogus-Block")
    errors = [p for p in _problems(bogus, "iris") if p[2] == "error"]
    assert errors[0][:2] == (lines.index("!assignments") + 1, 1) and "unknown section !Bogus-Block" in errors[0][3]
    stray = qpm.replace("std_me_pi = 0.1;", "std_me_pi = 0.1; nope = 1;")
    errors = [p for p in _problems(stray, "iris") if p[2] == "error"]
    assert [p[:2] for p in errors] == [place(stray, "std_me_pi = 0.1;", "nope")] and "not in !parameters" in errors[0][3]
    with pytest.raises(ModelError, match="kapp is not declared"):
        ModelSource(typo, "iris").build()


def test_a_long_name_with_parentheses_and_a_percent_sign():
    """Dynare's quoted long_name may hold parentheses and %: 'Output gap (% of potential)' is one decoration, not the
    end of one at the first ")" with the names after it swallowed (read_mod had "2 equations for 3 variables"), and the
    % inside it starts no comment (the editor's check had "p is not declared"). So the QPM toolkit's model, written to
    Dynare with its descriptions, reads back and is checked against the original: the same decision rules."""
    from test_iris import QPM_TOOLKIT_NK as qpm

    text = ("var y (long_name='Output gap (% of potential)') p $\\pi$ (long_name=\"prices (level)\");\nvarexo e;\n"
            "parameters a;\na = 0.5;\nmodel;\n  y = a*y(-1) + e;\n  p = y(+1);\nend;\n")
    plain = text.replace("Output gap (% of potential)", "Output gap").replace("prices (level)", "prices")
    read, without = read_mod(text), read_mod(plain)
    assert read.model.endogenous == without.model.endogenous == ("y", "p")
    assert read.labels == {"y": "Output gap (% of potential)", "p": "prices (level)"}
    found = ModelSource(text, "dynare").check()
    assert found.ok and not found.diagnostics, [str(d) for d in found.diagnostics]
    assert {s.name: s.description for s in found.symbols if s.role == "endogenous"}["y"] == "Output gap (% of potential)"
    assert {s.name: s.role for s in found.symbols} == {s.name: s.role for s in ModelSource(plain, "dynare").check().symbols}
    written = ModelSource(qpm, "iris").export("dynare")
    assert "long_name='Output gap (% of potential)'" in written.text
    assert written.checked and written.difference < 1e-10, written.notes
    dynare, iris = read_mod(written.text), read_iris(qpm)
    assert dynare.labels["y"] == "Output gap (% of potential)"
    a, b = dynare.solve().rules().to_frame().set_index(""), iris.solve().rules().to_frame().set_index("")
    np.testing.assert_allclose(a.loc[list(b.index), list(b.columns)].to_numpy(float), b.to_numpy(float), atol=1e-12)


def test_iris_reporting_equations_checked_and_written():
    """The editor's IRIS check takes !reporting_equations: each name on the left is a reported series (with its
    description), an undeclared name on the right is placed at its line and column; writing to IRIS writes them back as
    !reporting_equations, which read again give the same reported series and labels (and the parameter and shock-size
    descriptions); Dynare still says it has no equivalent."""
    from pymetrix.models.writers import export_model
    from test_iris import IRIS_REPORTING

    found = ModelSource(IRIS_REPORTING, "iris").check()
    assert found.ok and found.equations == 3
    reported = {s.name: s for s in found.symbols if s.role == "reported"}
    assert list(reported) == ["gap", "lvl", "chg", "zl"] and reported["gap"].description == "Gap, percent"
    assert reported["lvl"].used == (24,)  # chg = lvl - ... on line 24
    bad = [p for p in _problems(IRIS_REPORTING.replace("lvl = yp + g*y", "lvl = yq + g*y"), "iris") if p[2] == "error"]
    assert bad[0][:2] == (23, 11) and "yq is not declared" in bad[0][3]
    t = read_iris(IRIS_REPORTING)
    written = export_model(t, "iris")
    assert written.checked and written.difference < 1e-12 and not any("reported" in n for n in written.notes)
    assert "!reporting_equations" in written.text and '"Gap, percent" gap = 100*y;' in written.text
    assert '"Size of gap shocks" std_ey = 0.2' in written.text
    back = read_iris(written.text)
    assert [r.observable for r in back.measurement.reports] == ["gap", "lvl", "chg", "zl"]
    assert back.measurement.labels == t.measurement.labels and back.model.spec("rho").label == "Persistence of the gap"
    assert back.model.spec("sd_ey").label == "Size of gap shocks"
    assert any("reported series (gap, lvl, chg, zl) are not written: Dynare" in n for n in export_model(t, "dynare").notes)


SOURCES = {"pymetrix": NK, "dynare": NK_MOD, "iris": IRIS_OK}


@pytest.mark.parametrize("source", list(SOURCES))
@pytest.mark.parametrize("target", ["pymetrix", "dynare", "iris"])
def test_each_syntax_written_in_each_solves_the_same(source, target):
    translation = ModelSource(SOURCES[source], source).export(target)
    assert translation.checked and translation.difference < 1e-12, translation.notes
    assert translation.syntax == target and ModelSource(translation.text, target).check().ok
    assert "comments are not carried across" in translation.notes[0]
    if source == "dynare":
        assert any("commands are not written" in n for n in translation.notes)
    if target == "dynare" and source != "dynare":
        assert "varobs" in translation.text and any("measurement equations are written as equations" in n for n in translation.notes)


def test_harder_models():
    from pymetrix.models.writers import _compare, export_model

    rbc = ModelSource(IRIS_RBC, "iris")  # log variables, and equations with expressions on the left
    for target in ("pymetrix", "dynare", "iris"):
        written = rbc.export(target)
        assert written.checked and written.difference < 1e-12
        assert any("log variables" in n for n in written.notes)
    assert "1/exp(c) = beta/exp(c[+1])" in rbc.export("pymetrix").text
    zlb = export_model(templates.new_keynesian(lower_bound=0.0), "dynare")  # max: a function there, a constraint here
    assert zlb.checked and "max(" in zlb.text and any("occbin_constraints" in n for n in zlb.notes)
    lagged = ModelSource("parameters\n    a = 0.5\nshocks\n    e = 0.1\nequations\n    x = a*x[-1] + e\n    z = z[-1] + d(x) + dlog(w)\n"
                         "    w = exp(x)\n")
    dyn = lagged.export("dynare")
    assert dyn.checked and dyn.difference < 1e-12 and "x - x(-1) + log(w) - log(w(-1))" in dyn.text and "d(x)" not in dyn.text
    minus = ModelSource("shocks\n    e = 0.1\nequations\n    x = 0.5*x[-1] + e\n    z = z[-1] - d(x) - 2*dlog(x + 3)\n").export("iris")
    assert minus.checked and minus.difference < 1e-12 and "- (x - x{-1}) - 2*(log(x + 3) - log(x{-1} + 3))" in minus.text
    assert _compare(templates.new_keynesian(), templates.new_keynesian(beta=0.9), ["ey", "epi", "ei"]) > 1e-3  # a gap is seen
    with pytest.raises(ModelError, match="no equivalent in Dynare or IRIS"):
        ModelSource("shocks\n    e = 1\nequations\n    x = (x[-1] > 0) + e\n").export("dynare")
    from pymetrix import Equation, Model

    with pytest.raises(ModelError, match="linear form"):
        export_model(Model([Equation("x ~ x[-1]")], params={"const": 0.0, "x[-1]": 0.5}), "dynare")


def test_writing_from_the_editor_and_iris_in_a_step(tmp_path):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.commands import Session
    from pymetrix.commands.model_source import DefineModel
    from pymetrix.ui.model_editor_qt import ModelEditorWindow

    window = ModelEditorWindow(None, text=NK, name="nk")
    other = window.export("iris")
    assert other.syntax.currentData() == "iris" and other.editor.highlighter.syntax == "iris"
    assert other.sidecars.tabText(other.sidecars.currentIndex()) == "Translation" and other.last.ok
    assert other.editor.toPlainText() == ModelSource(NK).export("iris").text
    assert window.export("dynare") is not None and window.syntax.findData("iris") >= 0
    broken = ModelEditorWindow(None, text=NK.replace("kappa*y", "kapa*y"))
    assert broken.export("dynare") is None and "Could not write it" in broken.status.text()
    s = Session()
    s.run(DefineModel(IRIS_OK, syntax="iris", name="nk"))
    assert type(s["nk"]).__name__ == "Template" and "syntax='iris'" in s.script()


# --- phase 3: the equations in algebraic notation, and the solution check ---------------------------------------------

SIMPLE = '''parameters
    beta = 0.99
    kappa = 0.1
    phi = 1.5
shocks
    e = 1
equations
    y = y[+1] - (phi*pi - pi[+1]) + e
    pi = beta*pi[+1] + kappa*y
'''


def test_equations_in_algebraic_notation_and_latex():
    rows = ModelSource(NK).algebraic()
    assert [r.name for r in rows] == ["IS", "Phillips", "policy", "infl"]
    assert rows[0].text == "yₜ = yₜ₊₁ − (1 / σ) (iₜ − πₜ₊₁) + eyₜ"  # the division bracketed: not 1 / (σ (iₜ − πₜ₊₁))
    assert rows[1].text == "πₜ = β πₜ₊₁ + κ yₜ + epiₜ"
    assert rows[2].text == "iₜ = ρ iₜ₋₁ + (1 − ρ) (φ_π πₜ + φ_y yₜ) + eiₜ" and rows[2].variable == "i"
    assert rows[3].kind == "measurement" and rows[3].text == "inflₜ = 4 πₜ + ηₜ"
    tex = ModelSource(NK).latex()
    assert "y_t &= y_{t+1} - \\frac{1}{\\sigma} \\left(i_t - \\pi_{t+1}\\right) + \\mathit{ey}_t \\tag{IS} \\\\" in tex
    assert tex.count("\\begin{align}") == 2 and tex.count("\\end{align}") == 2 and "\\tag{infl}" in tex
    assert tex.count("\\left(") == tex.count("\\right)") and tex.count("{") == tex.count("}")
    zlb = "\n".join(r.text for r in __import__("pymetrix.models.writers", fromlist=["x"]).algebraic_equations(
        templates.new_keynesian(lower_bound=0.0), "latex"))
    assert "\\max\\left(" in zlb and "\\operatorname{max}" not in zlb
    # every syntax, the same equations: Dynare's and IRIS's text give the same notation as pymetrix's
    dyn = ModelSource(NK_MOD, "dynare").algebraic()
    assert dyn and all("ₜ" in r.text for r in dyn)
    assert {r.variable for r in ModelSource(IRIS_OK, "iris").algebraic() if r.kind == "equation"} == set(read_iris(IRIS_OK).model.endogenous)
    with pytest.raises(ModelError, match="errors"):
        ModelSource(NK.replace("kappa*y", "kapa*y")).algebraic()


def test_the_solution_check_against_the_roots_by_hand():
    beta, kappa = 0.99, 0.1
    for phi, verdict in ((1.5, "unique"), (0.5, "indeterminate")):  # the Taylor principle
        found = ModelSource(SIMPLE.replace("phi = 1.5", f"phi = {phi}")).check_solution()
        assert found.verdict == verdict and found.ok == (verdict == "unique") and found.predetermined == 0
        # z(t+1) = M z(t) for z = (y, pi), written out from the two equations
        M = np.array([[1 + kappa / beta, phi - 1 / beta], [-kappa / beta, 1 / beta]])
        assert np.allclose(sorted(abs(r) for r in found.roots), sorted(abs(np.linalg.eigvals(M))), atol=1e-12)
        assert found.stable == sum(abs(np.linalg.eigvals(M)) < 1) and found.steady == {"y": 0.0, "pi": 0.0}
        assert len(found.roots_table().rows) == 2 and found.summary().rows[0][0] == "y"
    explosive = ModelSource("parameters\n    a = 1.5\nshocks\n    e = 1\nequations\n    x = a*x[-1] + e\n").check_solution()
    assert explosive.verdict == "no stable solution" and explosive.predetermined == 1 and explosive.stable == 0
    assert np.isclose(abs(explosive.roots[0]), 1.5)
    missing = ModelSource("parameters\n    a = 1\nequations\n    x = log(x - a) + 5 + x[+1] - x[+1]\n").check_solution()
    assert missing.verdict == "no steady state" and not missing.ok and missing.steady is None
    nk = ModelSource(NK).check_solution()
    assert nk.ok and nk.stable == nk.predetermined == 1
    assert np.allclose(sorted(abs(r) for r in nk.roots if np.isfinite(abs(r))),
                       sorted(abs(templates.new_keynesian().solve().eigenvalues[np.isfinite(templates.new_keynesian().solve().eigenvalues)])))
    dyn = ModelSource(NK_MOD, "dynare").check_solution()
    steady = read_mod(NK_MOD).steady_state()
    assert dyn.ok and all(abs(dyn.steady[v] - steady[v]) < 1e-12 for v in dyn.steady)


def test_equations_and_solution_in_the_editor():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.model_editor_qt import ModelEditorWindow

    window = ModelEditorWindow(None, text=NK, name="nk")
    tabs = [window.sidecars.tabText(i) for i in range(window.sidecars.count())]
    assert tabs[2:] == ["Equations", "Solution"]
    assert window.equations.topLevelItemCount() == 0  # built only when shown
    window.sidecars.setCurrentIndex(2)
    assert window.equations.topLevelItemCount() == 4
    first = window.equations.topLevelItem(0)
    assert first.text(0) == "IS" and first.text(1).startswith("yₜ = yₜ₊₁")
    window._go(window.equations.topLevelItem(2))
    assert window.editor.textCursor().blockNumber() + 1 == NK.splitlines().index(
        next(line for line in NK.splitlines() if line.strip().startswith("policy:"))) + 1
    assert "\\tag{IS}" in window.copy_latex() and QtWidgets.QApplication.clipboard().text().startswith("% nk")
    found = window.check_solution()
    assert found.ok and window.sidecars.currentWidget() is window.solution_page
    assert window.steady.topLevelItemCount() == 3 and window.roots.topLevelItemCount() == len(found.roots_table().rows)
    assert "unique stable solution" in window.verdict.text()
    window.editor.setPlainText(NK.replace("phi_pi = 1.5", "phi_pi = 0.5"))
    window.refresh()
    assert "changed since the last check" in window.verdict.text()
    assert window.check_solution().verdict == "indeterminate" and "not unique" in window.verdict.text()
    window.editor.setPlainText(NK.replace("kappa*y", "kapa*y"))
    assert window.check_solution() is None and "errors" in window.verdict.text() and not window.solve_action.isEnabled()
    window.sidecars.setCurrentIndex(2)
    assert window.equations.topLevelItemCount() == 0 and "errors" in window.equations_note.text()


def test_check_solution_is_not_called_stale_until_the_text_changes():
    """Check solution right after the text was set (a file opened, typing just stopped): the check the editor still
    had due from the typing, 300 ms on, called the solution just found stale ("changed since the last check") though
    nothing had changed. A check now takes the place of one still due, and the Solution tab is stale only when the text
    or its syntax differs from the text it was found for."""
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.model_editor_qt import ModelEditorWindow

    window = ModelEditorWindow(None, text=NK, name="nk")
    window.editor.setPlainText(NK.replace("phi_pi = 1.5", "phi_pi = 1.6"))  # typed: a check is due
    assert window.timer.isActive()
    found = window.check_solution()
    assert found.ok and not window.timer.isActive()  # the check due was this one
    window.refresh()  # as the timer's check, had it fired: the text is the same
    assert "unique stable solution" in window.verdict.text()
    window.editor.setPlainText(NK.replace("phi_pi = 1.5", "phi_pi = 0.5"))
    window.refresh()
    assert "changed since the last check" in window.verdict.text()
