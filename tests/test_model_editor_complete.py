"""Regimes and shock correlations written in a model's text, and the model editor's completions and hover help.

Checked:
- a model text with a ``regimes`` section and ``corr`` lines gives exactly what the same model built in code gives
  with ``Regime(...)`` objects and ``shock_corr``: the path through the regimes (a solution's ``solve``), the regime
  filter's log likelihood on data simulated with correlated shocks, an estimate (the regimes inside the sample estimated
  with, those after it kept for the forecast) and the forecast;
- the text round-trips: built, written in pymetrix's syntax, it is the same text (but the writer's first line);
- each new kind of problem (regimes: unknown parameters and equations, bad periods, order, overlap, names, frequency,
  a replacement that changes the variable determined, unknown names in it; correlations: unknown shocks, a shock with
  itself, values outside (-1, 1), a pair given twice, correlations that are not positive definite together) is
  reported once, at its line and column;
- IRIS's ``corr_<a>__<b>`` parameters are read into the template's correlations, checked where they are declared, and
  written back; the Dynare text written from a model text with correlations reads back (``read_mod``) with the same;
- completions for a prefix hold the declared names and functions (with signatures on hover) and nothing from
  comments; hover help gives a parameter's value, description, estimation settings and lines; the same in the editor
  (offscreen Qt)."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, ModelSource, Regime, read_iris, read_mod, regime_filter
from pymetrix.estimation.priors import Uniform
from pymetrix.metadata import Param, Role
from pymetrix.models import MeasurementMap, Model, Scenario
from pymetrix.models.regimes import regime_loglike

TEXT = '''parameters
    sigma = 1
    beta = 0.99    "discount factor"
    kappa = 0.1
    rho = 0.8  bounds 0 1
    phi_pi = 1.5
    phi_y = 0.5
shocks
    ey = 0.5    "demand"
    epi = 0.3
    ei = 0.2
    corr ey, ei = 0.3  prior uniform(-0.9, 0.9)    "demand and policy"
    corr epi, ei = -0.2
equations
    IS: y = y[+1] - 1/sigma*(i - pi[+1]) + ey
    Phillips: pi = beta*pi[+1] + kappa*y + epi
    policy: i = rho*i[-1] + (1 - rho)*(phi_pi*pi + phi_y*y) + ei
measurement
    infl = 4*pi  error 0.1  "inflation, annual"
regimes
    2008Q1 post: phi_pi = 2  "post-crisis rule"
    2012Q1..2013Q4: rho = 0.95
    2030Q1 alt: policy: i = 0.9*i[-1] + 0.2*pi + ei
    2032Q1 hawk surprise: phi_pi = 3; policy: i = rho*i[-1] + (1 - rho)*phi_pi*pi + ei
'''
SHOCKS = ("ey", "epi", "ei")
SD = {"ey": 0.5, "epi": 0.3, "ei": 0.2}
CORR = {("ey", "ei"): 0.3, ("epi", "ei"): -0.2}
INSIDE = (Regime("2008Q1", {"phi_pi": 2.0}, name="post"), Regime("2012Q1", {"rho": 0.95}, end="2013Q4"))
AHEAD = (Regime("2030Q1", equations={"policy": "i = 0.9*i[-1] + 0.2*pi + ei"}, name="alt"),
         Regime("2032Q1", {"phi_pi": 3.0}, {"policy": "i = rho*i[-1] + (1 - rho)*phi_pi*pi + ei"}, name="hawk", expected=False))
MEASUREMENT = MeasurementMap(["infl = 4*pi"], errors={"infl": 0.1})


def _code_model():
    return Model([Equation("y ~ y[+1] - (1/sigma)*(i - pi[+1]) + ey", params=["sigma"], name="IS"),
                  Equation("pi ~ beta*pi[+1] + kappa*y + epi", params=["beta", "kappa"], name="Phillips"),
                  Equation("i ~ rho*i[-1] + (1 - rho)*(phi_pi*pi + phi_y*y) + ei", params=["rho", "phi_pi", "phi_y"], name="policy")],
                 params=dict(sigma=1.0, beta=0.99, kappa=0.1, rho=0.8, phi_pi=1.5, phi_y=0.5), roles={s: Role.SHOCK for s in SHOCKS},
                 specs=[Param("rho", bounds=(0, 1)), Param("corr_ey_ei", prior=Uniform(-0.9, 0.9))])


def _data(sol, periods=80, seed=3):
    """Observations of annual inflation from the base solution, the shocks drawn with the correlations."""
    rng = np.random.default_rng(seed)
    C = np.eye(3)
    for (a, b), r in CORR.items():
        C[SHOCKS.index(a), SHOCKS.index(b)] = C[SHOCKS.index(b), SHOCKS.index(a)] = r
    sd = np.array([SD[s] for s in SHOCKS])
    e = rng.standard_normal((periods, 3)) @ np.linalg.cholesky(sd[:, None] * C * sd[None, :]).T
    x, pi = np.zeros(len(sol.variables)), []
    for t in range(periods):
        x = sol.P @ x + sol.Q @ e[t]
        pi.append(x[list(sol.variables).index("pi")])
    infl = 4 * np.array(pi) + 0.1 * rng.standard_normal(periods)
    return Dataset.from_frame(pd.DataFrame({"infl": infl}, index=pd.period_range("2000Q1", periods=periods, freq="Q")))


def test_regimes_and_correlations_from_text_are_those_built_in_code():
    t = ModelSource(TEXT, name="nk").build()
    assert dict(t.shock_corr) == CORR
    assert [(r.start, r.end, r.name, r.expected, dict(r.params), dict(r.equations)) for r in t.regimes] == \
        [(r.start, r.end, r.name, r.expected, dict(r.params), dict(r.equations)) for r in (*INSIDE, *AHEAD)]
    assert t.regimes[0].description == "post-crisis rule"
    assert t.model.spec("corr_ey_ei").label == "demand and policy" and isinstance(t.model.spec("corr_ey_ei").prior, Uniform)
    assert t.system().shock_corr == CORR
    code = _code_model().linearize(shocks=list(SHOCKS)).solve()
    # the path through the regimes: the template's solution follows its own; the code's is given them
    span, sc = ("2025Q1", "2036Q4"), Scenario(set={"ey": {"2025Q1": -1.0}, "ei": {"2031Q1": 0.5}})
    by_text = t.solve().solve(None, span, sc).frame.loc["2025Q1":, ["y", "pi", "i"]].to_numpy(float)
    by_code = code.solve(None, span, sc, regimes=[*INSIDE, *AHEAD]).frame.loc["2025Q1":, ["y", "pi", "i"]].to_numpy(float)
    assert np.max(np.abs(by_text - by_code)) < 1e-12 and np.max(np.abs(by_code)) > 0.1
    assert not np.allclose(by_code, code.solve(None, span, sc).frame.loc["2025Q1":, ["y", "pi", "i"]].to_numpy(float))  # the regimes matter
    # the likelihood on data simulated with correlated shocks, through the regimes inside the sample
    data = _data(code)
    system = code.system(MEASUREMENT, SD, shock_corr=CORR)
    y, index = system.observations(data, None)
    by_hand = regime_loglike(code, system, INSIDE, y, index)
    assert regime_filter(t, data).loglike == pytest.approx(by_hand, abs=1e-9)
    assert by_hand != pytest.approx(regime_loglike(code, code.system(MEASUREMENT, SD), INSIDE, y, index), abs=1e-3)  # corr matters
    # an estimate: the template's regimes inside the sample estimated with, those after it kept for the forecast
    fit_text = t.estimate(data, ["rho"], cov="none")
    fit_code = _code_model().estimate(data, MEASUREMENT, list(SHOCKS), SD, ["rho"], cov="none", regimes=INSIDE, shock_corr=CORR)
    assert fit_text.stats["loglike"] == pytest.approx(fit_code.stats["loglike"], abs=1e-8)
    assert fit_text.params["rho"] == pytest.approx(fit_code.params["rho"], abs=1e-8)
    assert fit_text.ahead == AHEAD and fit_code.ahead == ()
    path_text = fit_text.forecast(("2020Q1", "2034Q4")).frame.loc["2020Q1":, ["y", "pi", "i", "infl"]]
    path_code = fit_code.forecast(("2020Q1", "2034Q4"), regimes=AHEAD).frame.loc["2020Q1":, ["y", "pi", "i", "infl"]]
    assert np.max(np.abs(path_text.to_numpy(float) - path_code.to_numpy(float))) < 1e-10
    assert "Regimes after the sample" in "\n".join(fit_text.summary().notes)


def test_the_text_round_trips_through_the_writer():
    written = ModelSource(TEXT).export("pymetrix")
    assert written.text.split("\n", 1)[1] == TEXT and written.checked
    assert not any("correlations" in n or "regimes" in n for n in written.notes)
    for syntax in ("dynare", "iris"):  # no place for regimes there: said so
        notes = ModelSource(TEXT).export(syntax).notes
        assert any(n.startswith("the regimes (post, 2012Q1, alt, hawk) are not written") for n in notes)


def _plant(text, old, new):
    assert old in text
    return text.replace(old, new)


BASE = TEXT.split("regimes\n")[0] + "regimes\n"
CASES = [  # what is planted, the line (its text) and the token the problem is at, and what it says
    (BASE + "    2008Q1: phi_pk = 2\n", "phi_pk", "phi_pk is not a parameter; did you mean 'phi_pi'"),
    (BASE + "    2008Q5: rho = 0.9\n", "2008Q5", "'2008Q5' is not a period"),
    (BASE + "    2008Q1: rho = 0.9\n    2007Q1: rho = 0.7\n", "2007Q1", "regimes go in order of their starts"),
    (BASE + "    2008Q1..2009Q4: rho = 0.9\n    2009Q1: rho = 0.7\n", "2009Q1:", "before the temporary regime before it ends"),
    (BASE + "    2008Q1 calm: rho = 0.9\n    2009Q1 calm: rho = 0.7\n", "calm: rho = 0.7", "two regimes are called 'calm'"),
    (BASE + "    2008Q1: polcy: i = 0\n", "polcy", "polcy is not the name of an equation; did you mean 'policy'"),
    (BASE + "    2008Q1: policy: y = 0.5*i\n", "y = 0.5", "the equation policy determines i"),
    (BASE + "    2008Q1: policy: i = 0.5*i[-1] + gamma*pi\n", "gamma", "gamma is not a variable, shock or parameter"),
    (BASE + "    2009Q1..2008Q4: rho = 0.9\n", "2008Q4", "ends (2008Q4) before it starts"),
    (BASE + "    2008Q1: rho = 0.9\n    2009-03: rho = 0.7\n", "2009-03", "not of the first regime's frequency"),
    (BASE + "    2008Q1: rho 0.9\n", "rho 0.9", "a regime gives parameters values as name = value"),
    (BASE + "    2008Q1: rho = 0.9, rho = 0.7\n", "rho = 0.7", "rho is given twice in this regime"),
    (_plant(TEXT, "corr epi, ei", "corr epi, eii"), "eii", "eii is not a shock (declared in shocks); did you mean 'ei'"),
    (_plant(TEXT, "corr epi, ei", "corr epi, epi"), "epi = -0.2", "a correlation is between two different shocks"),
    (_plant(TEXT, "corr epi, ei = -0.2", "corr epi, ei = -1.2"), "-1.2", "a correlation is between -1 and 1"),
    (_plant(TEXT, "corr epi, ei = -0.2", "corr ei, ey = 0.1"), "corr ei, ey", "the correlation of ei and ey is given twice"),
    (_plant(TEXT, "corr epi, ei = -0.2", "corr ey, epi = 0.6\n    corr epi, ei = 0.6\n    corr ey, ei = -0.6")
     .replace("    corr ey, ei = 0.3  prior uniform(-0.9, 0.9)    \"demand and policy\"\n", ""),
     "corr ey, ei = -0.6", "not a valid correlation matrix"),
]


@pytest.mark.parametrize("text, token, message", CASES)
def test_each_new_problem_once_at_its_line_and_column(text, token, message):
    found = ModelSource(text).check()
    lines = text.split("\n")
    where = max(n for n, line in enumerate(lines, 1) if token in line)
    column = lines[where - 1].index(token) + 1
    assert [(d.line, d.column) for d in found.errors] == [(where, column)], found.errors
    assert message in found.errors[0].message


def test_regime_and_correlation_values_from_parameters_and_unused_ones():
    text = _plant(TEXT, "    phi_y = 0.5\n", "    phi_y = 0.5\n    lam = 0.5\n")
    text = _plant(text, "corr epi, ei = -0.2", "corr epi, ei = -lam/2")
    text = _plant(text, "2030Q1 alt: policy: i = 0.9*i[-1] + 0.2*pi + ei", "2030Q1 alt: policy: i = lam*i[-1] + 0.2*pi + ei")
    found = ModelSource(text).check()
    assert found.ok and not found.diagnostics, found.diagnostics  # lam: used by a regime's equation (and a correlation's value)
    t = ModelSource(text).build()
    assert t.shock_corr[("epi", "ei")] == -0.25 and t.regimes[2].params == {"lam": 0.5}  # given its declared value
    corr = next(s for s in found.symbols if s.name == "corr_ey_ei")
    assert corr.role == "correlation" and corr.value == 0.3 and "uniform" in corr.estimation.lower()
    bad = _plant(TEXT, "2008Q1 post: phi_pi = 2", "2008Q1 post: phi_pi = 2, lam = 1")
    assert "lam is not a parameter" in ModelSource(bad).check().errors[0].message


IRIS = '''!transition_variables
    y, pi, i
!transition_shocks
    ey, epi, ei
!parameters
    sigma = 1, beta = 0.99, kappa = 0.1, rho = 0.8, phi_pi = 1.5, phi_y = 0.5
    std_ey = 0.5, std_epi = 0.3, std_ei = 0.2
    "demand and policy" corr_ey__ei = 0.3
    corr_ei__epi = -0.2
!transition_equations
    y = y{+1} - 1/sigma*(i - pi{+1}) + ey;
    pi = beta*pi{+1} + kappa*y + epi;
    i = rho*i{-1} + (1 - rho)*(phi_pi*pi + phi_y*y) + ei;
'''


def test_iris_correlations_read_checked_and_written():
    nk = read_iris(IRIS)
    assert dict(nk.shock_corr) == {("ey", "ei"): 0.3, ("epi", "ei"): -0.2}
    assert nk.model.spec("corr_ey_ei").label == "demand and policy"
    found = ModelSource(IRIS, "iris").check()
    assert found.ok and not found.diagnostics, found.diagnostics  # no "not used" warnings for them
    assert {s.name: s.value for s in found.symbols}["corr_ei__epi"] == -0.2
    out = ModelSource(IRIS, "iris").export("iris")
    assert '    "demand and policy" corr_ey__ei = 0.3' in out.text and "    corr_epi__ei = -0.2" in out.text and out.checked
    assert dict(read_iris(out.text).shock_corr) == dict(nk.shock_corr)
    assert dict(ModelSource(TEXT).build().shock_corr) == dict(read_iris(ModelSource(TEXT).export("iris").text).shock_corr)
    for planted, token, message in (("corr_ei__epi = -0.2", "corr_ei__ez", "not a transition shock"),
                                    ("corr_ei__epi = -0.2", "corr_ei__epi = -1.5", "between -1 and 1")):
        text = IRIS.replace(planted, token if "=" in token else token + " = -0.2")
        lines = text.split("\n")
        where = next(n for n, line in enumerate(lines, 1) if token.split(" ")[0] in line)
        errors = ModelSource(text, "iris").check().errors
        assert [(d.line, d.column) for d in errors] == [(where, lines[where - 1].index(token.split(" ")[0]) + 1)], errors
        assert message in errors[0].message


def test_dynare_text_written_from_correlations_reads_back_the_same():
    out = ModelSource(TEXT).export("dynare")
    assert "  corr ey, ei = 0.3;" in out.text and "  corr epi, ei = -0.2;" in out.text and out.checked
    dyn = read_mod(out.text)
    assert dict(dyn.shock_corr) == CORR
    assert dyn.priors["corr_ey_ei"].low == -0.9 and dyn.priors["corr_ey_ei"].high == 0.9  # the prior as well


COMMENTED = TEXT + "# phi_comment and normtext\n"


def test_completions_and_hover_help():
    src = ModelSource(COMMENTED)
    lines = COMMENTED.split("\n")
    policy = next(n for n, line in enumerate(lines, 1) if line.startswith("    policy:"))
    at = lines[policy - 1].index("phi_pi") + 4  # the cursor after "phi"
    assert src.completions(policy, at) == ["phi_pi", "phi_y", "Phillips"]
    norm = ModelSource(COMMENTED.replace("+ ei\nmeasurement", "+ ei + norm\nmeasurement"))
    nlines = norm.text.split("\n")
    nline = next(n for n, line in enumerate(nlines, 1) if line.endswith("+ norm"))
    found = norm.completions(nline, len(nlines[nline - 1]) + 1)
    assert {"normcdf", "normpdf", "norminv", "normal"} <= set(found) and "normtext" not in found
    last = len(lines) - 1  # the comment line
    assert src.completions(last, lines[last - 1].index("phi_comment") + 4) == []  # nothing inside a comment
    assert all("comment" not in w and "normtext" not in w for w in src.completions(policy, 5, None) + src.completions(1, 1))
    shocks = next(n for n, line in enumerate(lines, 1) if line.startswith("    corr ey"))
    assert src.completions(shocks, 11)[:3] == ["ey", "epi", "ei"]  # "    corr e": the shocks first
    rho = src.hover(policy, lines[policy - 1].index("rho") + 2)
    assert rho.startswith("rho: parameter = 0.8") and "estimation: bounds 0 to 1" in rho and f"used on lines {policy}" in rho
    beta = src.hover(3, 6)
    assert beta.splitlines()[:2] == ["beta: parameter = 0.99", "discount factor"]
    assert src.hover(9, 5).startswith("ey: shock, standard deviation 0.5\ndemand")
    assert src.hover(3, 22) == ""  # inside the description
    call = ModelSource(TEXT.replace("(1 - rho)*(phi_pi", "(1 - rho)*(normcdf(phi_pi)"))
    cline = call.text.split("\n")[policy - 1]
    assert call.hover(policy, cline.index("normcdf") + 3) == "normcdf(x, mu = 0, sigma = 1): the normal cumulative distribution function"
    assert call.hover(policy, cline.index("i =") + 1).startswith("i: endogenous variable\ndetermined on line")
    dyn = ModelSource("var y;\nvarexo e;\nparameters rho;\nrho = 0.8;\nmodel;\n  y = rho*y(-1) + e;\nend;\n", "dynare")
    assert "varexo" in dyn.completions(2, 4) and dyn.hover(6, 8).startswith("rho: parameter = 0.8")
    iris = ModelSource(IRIS, "iris")
    assert "!transition_equations" in iris.completions(10, 6)


def test_completion_and_hover_in_the_editor():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QTextCursor
    from PySide6.QtTest import QTest

    from pymetrix.ui.model_editor_qt import ModelEditorWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = ModelEditorWindow(text=COMMENTED)
    editor = window.editor
    cursor = editor.textCursor()
    cursor.movePosition(QTextCursor.MoveOperation.End)
    editor.setTextCursor(cursor)
    QTest.keyClicks(editor, "# no ph")  # in a comment: nothing offered
    assert not editor.completer.popup().isVisible() and editor.completions() == []
    QTest.keyClick(editor, Qt.Key.Key_Return)
    QTest.keyClick(editor, Qt.Key.Key_Space, Qt.KeyboardModifier.ControlModifier)  # Ctrl+Space: everything
    assert editor.completer.popup().isVisible() and editor.completer.completionCount() > 20
    editor.completer.popup().hide()
    QTest.keyClicks(editor, "x = p")  # one character: not yet
    assert not editor.completer.popup().isVisible()
    QTest.keyClicks(editor, "h")  # two characters: offered at once
    assert editor.completer.popup().isVisible() and editor.completions() == ["phi_pi", "phi_y", "Phillips"]
    editor._insert_completion("Phillips")  # the word typed replaced, whatever its case
    assert editor.toPlainText().endswith("x = Phillips")
    QTest.keyClicks(editor, " + movavg(")
    assert editor.signature() == "movavg(x, n): the average over a trailing window of n periods"
    window.refresh()
    lines = editor.toPlainText().split("\n")
    rho = next(n for n, line in enumerate(lines, 1) if line.startswith("    rho"))
    assert editor.help_at(rho, 6).startswith("rho: parameter = 0.8\nestimation: bounds 0 to 1")
    bad = [d for d in window.last.diagnostics if d.severity == "error"][0]  # the line typed: an underlined problem
    assert editor.help_at(bad.line, bad.column) == bad.message
    window.close()
    app.processEvents()
