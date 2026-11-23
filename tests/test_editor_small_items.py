"""Four small items of the editors and the expression language, each against an independent reference.

What is checked:

* the model editor's impulse-response preview (Solution tab, after Check solution): the responses drawn are exactly
  ``LinearSolution.irf`` of the model's solution with its shock sizes and correlations (correlated shocks
  orthogonalised by Cholesky), for the shock chosen and the variables ticked (the first six endogenous by default);
  redrawn when the shock changes; none when the solution is not unique; with regimes, the base regime's (said so);
* the Model > Matrix > Expression… dialog checks the expression as it is typed: unknown names (with suggestions),
  forbidden syntax and parse errors (with the column) shown, OK disabled while there is an error, and the result's
  shape and names shown when it evaluates cheaply;
* ``poissinv`` and ``binoinv`` (MATLAB: the smallest integer k with cdf(k) >= p) against scipy.stats ``ppf`` for many
  p, 0 and 1 and the edges included, and refused in linearisation (no derivative);
* centred and expanding moving windows (``movavg(x, 5, "centred")``, ``movavg(x, "expanding")`` and the other moving
  functions) against pandas ``rolling(center=True)`` and ``expanding()``, per entity in a panel, missing at the ends
  until the window is full; centred windows read as leads by the model compiler, expanding ones refused in models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

NK_CORR = '''parameters
    beta = 0.99
    sigma = 1
    kappa = 0.1
    phi_pi = 1.5
    phi_y = 0.5
    rho = 0.8
shocks
    ey = 0.5
    epi = 0.3
    ei = 0.2
    corr ey, ei = 0.4
equations
    IS: y = y[+1] - (1/sigma)*(i - pi[+1]) + ey
    Phillips: pi = beta*pi[+1] + kappa*y + epi
    policy: i = rho*i[-1] + (1 - rho)*(phi_pi*pi + phi_y*y) + ei
'''


@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("matplotlib")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_irf_preview_is_the_solutions_irf(qapp):
    from pymetrix.models.source import ModelSource
    from pymetrix.ui.model_editor_qt import ModelEditorWindow

    window = ModelEditorWindow(text=NK_CORR)
    found = window.check_solution()
    assert found.ok and window.irf_shock.count() == 3
    template = ModelSource(NK_CORR).build()
    reference = template.solve().irf(20, shock_sd=template.shock_sd, shock_corr=template.shock_corr)
    assert "orthogonalised" in reference.identification
    for k, shock in enumerate(("ey", "epi", "ei")):
        window.irf_shock.setCurrentIndex(k)  # redrawn on the change
        shown = window.irf_shown()
        assert list(shown) == ["y", "pi", "i"]
        for name, values in shown.items():
            assert len(values) == 21 and np.allclose(values, reference[name, shock].to_numpy(), atol=1e-12)
        assert window.irf_canvas.figure.axes and f"Responses to {shock}" in window.irf_canvas.figure._suptitle.get_text()
    # untick a variable: the figure shows the others
    window.irf_variables.item(1).setCheckState(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.CheckState.Unchecked)
    assert list(window.irf_shown()) == ["y", "i"] and len(window.irf_canvas.figure.axes) == 2
    # the correlation matters: the response of i to ey differs from the uncorrelated one
    plain = template.solve().irf(20, shock_sd=template.shock_sd)
    assert not np.allclose(plain["i", "ey"], reference["i", "ey"])
    # not unique: no responses
    window.editor.setPlainText(NK_CORR.replace("phi_pi = 1.5", "phi_pi = 0.5"))
    found = window.check_solution()
    assert not found.ok and window.irf_shown() == {} and window.irf_shock.count() == 0
    assert "no unique" in window.irf_note.text()
    with pytest.raises(Exception, match="no unique stable solution"):
        found.irf()


def test_irf_preview_with_regimes_is_the_base_regimes(qapp):
    from pymetrix.ui.model_editor_qt import ModelEditorWindow

    text = NK_CORR + "regimes\n    2025Q1 hawk: phi_pi = 3\n"
    window = ModelEditorWindow(text=text)
    found = window.check_solution()
    assert found.ok and found.regimes == 1 and "base regime" in window.irf_note.text()
    base = ModelEditorWindow(text=NK_CORR)
    base.check_solution()
    for a, b in zip(window.irf_shown().values(), base.irf_shown().values()):
        assert np.allclose(a, b)


def test_irf_panel_text_in_the_apps_text_size(qapp):
    """Each panel's title is 8 points and its tick labels 7, times the app's text scale, and the figure's title 9 times
    it (the loop over panels once reused the scale's name, giving sizes 0, 8, 16... by panel)."""
    from pathlib import Path

    from pymetrix.ui.chart_look import look
    from pymetrix.ui.model_editor_qt import ModelEditorWindow

    text = (Path(__file__).parent / "fixtures" / "dynare" / "rbc_order2.mod").read_text()
    window = ModelEditorWindow(text=text, syntax="dynare")
    assert window.check_solution().ok
    window.draw_irf()
    scale = look(window)[1].text_scale
    figure = window.irf_canvas.figure
    assert len(figure.axes) > 1
    assert [ax.title.get_fontsize() for ax in figure.axes] == [pytest.approx(8 * scale)] * len(figure.axes)
    for ax in figure.axes:
        assert all(label.get_fontsize() == pytest.approx(7 * scale) for label in ax.get_xticklabels() + ax.get_yticklabels())
    assert figure._suptitle.get_fontsize() == pytest.approx(9 * scale)
    window.close()


def test_matrix_expression_checked_as_typed(qapp):
    from pymetrix.commands import DefineMatrix, EnterMatrix, Session
    from pymetrix.commands.matrices import live_check
    from pymetrix.ui.qt import command_dialog

    session = Session()
    session.run(EnterMatrix("1\t0.5\n0.5\t2", rows="a, b", cols="a, b", name="G"))
    session.run(EnterMatrix("1 2", name="g"))
    dialog = command_dialog(DefineMatrix, session)
    ok = dialog.findChild(__import__("PySide6.QtWidgets", fromlist=["x"]).QDialogButtonBox).button(
        __import__("PySide6.QtWidgets", fromlist=["x"]).QDialogButtonBox.StandardButton.Ok)
    expr, name = dialog.editors["expression"].widget, dialog.editors["name"].widget
    assert dialog.expression_timer.interval() == 300 and dialog.expression_timer.isSingleShot()
    cases = [
        ("G.inv()", True, "a 2 x 2 matrix; rows a, b; columns a, b"),
        ("Gg.inv()", False, "no object named 'Gg' (did you mean "),
        ("G @@ G", False, "not a Python expression (invalid syntax at column 4)"),
        ("lambda: G", False, "lambda is not allowed in a matrix expression"),
        ("G.to_csv('x')", False, "to_csv reads or writes files"),
        ("G.__class__", False, "attributes starting with an underscore"),
        ("import os", False, "imports are not allowed"),
        ("G @ g", False, "against"),  # evaluated: the shapes do not conform
        ("G.det()", False, "gives a number, not a matrix"),
        ("G.svd()", True, "a decomposition: SVD (U, s, V)"),
        ("np.zeros((1000, 1000, 1000))", True, "not evaluated while typing: it may make a large array"),
        ("Matrix.identity(2 ** 40)", True, "not evaluated while typing"),
    ]
    for text, good, message in cases:
        expr.setText(text)
        assert dialog.expression_timer.isActive()  # checked a moment after typing stops, not at once
        dialog.check_expression()
        assert ok.isEnabled() is good and message in dialog.expression_check.text(), (text, dialog.expression_check.text())
        found = live_check(text, "M", session)
        assert found.ok is good and (good or found.problem in ok.toolTip())
    expr.setText("G.T")
    name.setText("2x")
    dialog.check_expression()
    assert not ok.isEnabled() and "not a Python name" in dialog.expression_check.text()
    name.setText("M")
    dialog.check_expression()
    assert ok.isEnabled()
    # nothing ran: the session holds only the two matrices entered
    assert set(session.objects) == {"G", "g"}
    # large inputs: checked, not evaluated
    session.run(EnterMatrix("\n".join("\t".join("1" for _ in range(210)) for _ in range(210)), name="Big"))
    found = live_check("Big.T @ Big", "M", session)
    assert found.ok and found.shape is None and "44,100 numbers" in found.skipped


# poissinv and binoinv: MATLAB's smallest integer k with cdf(k) >= p, against scipy's ppf

PROBS = np.r_[0.0, 1e-300, 1e-12, np.linspace(0.0005, 0.9995, 400), 1 - 1e-12, 1.0]


def _eval(text, x):
    from pymetrix.expr import evaluate, parse_expression

    return np.asarray(evaluate(parse_expression(text), lambda name, lag: x), dtype=float)


@pytest.mark.parametrize("fn, params, dist", [
    ("poissinv", (0.3,), lambda: stats().poisson(0.3)), ("poissinv", (4.0,), lambda: stats().poisson(4.0)),
    ("poissinv", (37.5,), lambda: stats().poisson(37.5)),
    ("binoinv", (1.0, 0.5), lambda: stats().binom(1, 0.5)), ("binoinv", (15.0, 0.2), lambda: stats().binom(15, 0.2)),
    ("binoinv", (40.0, 0.97), lambda: stats().binom(40, 0.97)), ("binoinv", (200.0, 0.01), lambda: stats().binom(200, 0.01)),
])
def test_integer_quantiles_against_scipy(fn, params, dist):
    d = dist()
    got = _eval(f"{fn}(x, {', '.join(map(repr, params))})", PROBS)
    want = np.maximum(d.ppf(PROBS), 0.0)  # scipy's ppf(0) is -1, below the support; MATLAB's is 0
    np.testing.assert_array_equal(got, want)
    assert got[0] == 0.0 and got[-1] == (np.inf if fn == "poissinv" else params[0])
    # the definition: cdf(k) >= p, and cdf(k - 1) < p (k the smallest)
    finite = np.isfinite(got)
    assert np.all(d.cdf(got[finite]) >= PROBS[finite] * (1 - 1e-12))
    inner = finite & (PROBS > 0) & (PROBS < 1 - 1e-9)  # nearer 1 the cdf rounds to 1 below the quantile, as in scipy
    assert np.all(d.cdf(got[inner] - 1) < PROBS[inner])
    # back from the cdf at each integer of the support: the integer itself
    ks = np.arange(0, int(d.ppf(0.999)) + 1, dtype=float)
    np.testing.assert_array_equal(_eval(f"{fn}({fn.replace('inv', 'cdf')}(x, {', '.join(map(repr, params))}), "
                                        f"{', '.join(map(repr, params))})", ks), ks)


def stats():
    return pytest.importorskip("scipy.stats")


def test_integer_quantiles_edges_and_invalid_values():
    p = np.array([-0.1, 1.1, np.nan, 0.0, 0.5, 1.0])
    np.testing.assert_array_equal(_eval("poissinv(x, 0)", p), [np.nan, np.nan, np.nan, 0.0, 0.0, 0.0])  # all mass at 0
    np.testing.assert_array_equal(_eval("poissinv(x, -1)", p), [np.nan] * 6)
    np.testing.assert_array_equal(_eval("binoinv(x, 10, 0)", p), [np.nan, np.nan, np.nan, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(_eval("binoinv(x, 10, 1)", p), [np.nan, np.nan, np.nan, 0.0, 10.0, 10.0])
    assert np.isnan(_eval("binoinv(x, 10.5, 0.3)", np.array([0.5]))).all()  # n must be an integer
    assert np.isnan(_eval("binoinv(x, 10, 1.3)", np.array([0.5]))).all()


def test_integer_quantiles_refuse_linearisation_and_are_listed():
    from pymetrix.expr import Name, diff, parse_expression
    from pymetrix.expr.functions import DISTRIBUTIONS, signature
    from pymetrix.models.solvers import ModelError
    from pymetrix.models.source import ModelSource, function_help

    for text, wrt in (("poissinv(x, 3)", "x"), ("poissinv(0.5, x)", "x"), ("binoinv(x, 10, 0.3)", "x"), ("binoinv(0.4, 10, x)", "x")):
        with pytest.raises(Exception, match="no derivative and cannot be linearised"):
            diff(parse_expression(text), Name(wrt))
    assert diff(parse_expression("poissinv(0.5, 3) * x"), Name("x")) is not None  # a constant factor is fine
    assert signature("poissinv") == "poissinv(p, lambda)" and signature("binoinv") == "binoinv(p, n, p)"
    assert "smallest integer k" in function_help("poissinv") and {"poissinv", "binoinv"} <= set(DISTRIBUTIONS)
    text = "parameters\n    rho = 0.5\nshocks\n    e\nequations\n    y = rho*y[-1] + 0.1*poissinv(normcdf(y[-1]), 2) + e\n"
    src = ModelSource(text)
    assert "poissinv" in src.completions(6, 30) or "poissinv" in [w for w in src.completions(6, 1)]
    with pytest.raises((ModelError, Exception), match="no derivative"):
        src.build().model.linearize(shocks=["e"])


# centred and expanding windows, against pandas


def _panel():
    from pymetrix import Dataset

    rng = np.random.default_rng(11)
    rows = []
    for country, n in [("AUS", 31), ("NZL", 24), ("CAN", 9)]:
        for period in pd.period_range("2000Q1", periods=n, freq="Q"):
            rows.append({"country": country, "time": period, "x": rng.standard_normal(), "z": rng.standard_normal()})
    frame = pd.DataFrame(rows).set_index(["country", "time"])
    frame.iloc[40, 0] = np.nan  # a missing value in NZL's x
    return Dataset.from_frame(frame.reset_index(), time="time", entity="country"), frame


CENTRED = [
    ("cmovavg(x, 5)", lambda f: f["x"].rolling(5, center=True).mean()),
    ("cmovavg(x, 4)", lambda f: f["x"].rolling(4, center=True).mean()),  # even: one more lag than leads
    ("cmovsum(x, 3)", lambda f: f["x"].rolling(3, center=True).sum()),
    ("cmovsd(x, 6)", lambda f: f["x"].rolling(6, center=True).std()),
    ("cmovvar(x, 7)", lambda f: f["x"].rolling(7, center=True).var()),
    ("cmovmin(x, 3)", lambda f: f["x"].rolling(3, center=True).min()),
    ("cmovmax(x, 4)", lambda f: f["x"].rolling(4, center=True).max()),
    ("cmovcov(x, z, 5)", lambda f: f["x"].rolling(5, center=True).cov(f["z"])),
    ("cmovcorr(x, z, 6)", lambda f: f["x"].rolling(6, center=True).corr(f["z"])),
    ("cmovbeta(z, x, 5)", lambda f: f["z"].rolling(5, center=True).cov(f["x"]) / f["x"].rolling(5, center=True).var()),
]


def _beta(f, y, x):
    both = f[x].notna() & f[y].notna()
    a, b = f[y].where(both), f[x].where(both)
    return a.expanding().cov(b) / b.expanding().var()


EXPANDING = [
    ("cumavg(x)", lambda f: f["x"].expanding().mean()),
    ("cumsum(x)", lambda f: f["x"].expanding().sum()),
    ("cumsd(x)", lambda f: f["x"].expanding().std()),
    ("cumvar(x)", lambda f: f["x"].expanding().var()),
    ("cummin(x)", lambda f: f["x"].expanding().min()),
    ("cummax(x)", lambda f: f["x"].expanding().max()),
    ("cumcov(x, z)", lambda f: f["x"].expanding().cov(f["z"])),
    ("cumcorr(x, z)", lambda f: f["x"].expanding().corr(f["z"])),
    ("cumbeta(z, x)", lambda f: _beta(f, "z", "x")),
    ("cumavg(x[-1])", lambda f: f["x"].shift(1).expanding().mean()),
]


@pytest.mark.parametrize("text, reference", CENTRED + EXPANDING, ids=[c[0] for c in CENTRED + EXPANDING])
def test_centred_and_expanding_windows_against_pandas_per_entity(text, reference):
    from pymetrix.expr import parse_expression

    panel, frame = _panel()
    got = panel.eval(text)
    for country in ["AUS", "NZL", "CAN"]:
        want = reference(frame.xs(country, level="country"))
        mine = got.xs(country, level="country")
        np.testing.assert_allclose(mine.to_numpy(), want.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=f"{text} {country}")
        assert mine.isna().sum() == want.isna().sum()
    if text.startswith("cmov"):  # missing at both ends until the window is full
        n = int(text.rstrip(")").rsplit(",", 1)[1])
        aus = got.xs("AUS", level="country")
        assert aus.iloc[: n // 2].isna().all() and aus.iloc[len(aus) - (n - 1) // 2:].isna().all()
        assert aus.iloc[n // 2: len(aus) - (n - 1) // 2].notna().all()
    e = parse_expression(text)
    assert str(e) == text and parse_expression(str(e)) == e
    # a single time series, too
    one = __import__("pymetrix").Dataset.from_frame(frame.xs("AUS", level="country"))
    np.testing.assert_allclose(one.eval(text).to_numpy(), reference(frame.xs("AUS", level="country")).to_numpy(),
                               rtol=1e-9, atol=1e-12, equal_nan=True)


def test_centred_windows_are_leads_with_derivatives():
    from pymetrix.expr import Name, diff, evaluate, parse_expression
    from pymetrix.expr.compile import compile_exprs
    from pymetrix.notation import expression

    e = parse_expression("cmovcorr(x, z, 4) + cmovbeta(z, x[-1], 3) + cmovsd(x, 5) + cmovavg(x, 3)")
    assert {n.lag for n in e.refs() if n.id == "x"} == {2, 1, 0, -1, -2}
    rng = np.random.default_rng(2)
    point = {(n.id, n.lag): rng.standard_normal() for n in e.refs()}
    f = lambda p: float(evaluate(e, lambda name, lag: p[(name, lag)]))  # noqa: E731
    for ref in e.refs():
        h = 1e-6
        up, down = dict(point), dict(point)
        up[(ref.id, ref.lag)] += h
        down[(ref.id, ref.lag)] -= h
        exact = float(evaluate(diff(e, Name(ref.id, ref.lag)), lambda name, lag: point[(name, lag)]))
        assert exact == pytest.approx((f(up) - f(down)) / (2 * h), rel=1e-5, abs=1e-7), str(ref)
    compiled = compile_exprs([e])
    assert float(compiled([point[r] for r in compiled.refs])[0]) == pytest.approx(f(point), rel=1e-12)
    with pytest.raises(Exception, match="cannot be differentiated"):
        diff(parse_expression("cmovmax(x, 3)"), Name("x", 1))
    assert expression(parse_expression("cmovavg(x, 4)")).startswith("cmovavg(x")


def test_centred_windows_in_models_are_leads_and_expanding_ones_refused():
    from pymetrix.expr import Name, diff, parse_expression
    from pymetrix.models.solvers import ModelError
    from pymetrix.models.source import ModelSource, function_help

    text = ("parameters\n    rho = 0.5\nshocks\n    e\nequations\n"
            "    y = rho*y[-1] + 0.2*{} + e\n")
    centred = ModelSource(text.format("cmovavg(y, 3)"))
    found = centred.check()
    assert found.ok and found.max_lead == 1 and found.max_lag == 1
    solved = centred.check_solution()
    assert solved.verdict in ("unique", "indeterminate", "no stable solution")
    trailing = ModelSource(text.format("movavg(y, 3)")).check()
    assert trailing.max_lead == 0
    bad = ModelSource(text.format("cumavg(y)"))
    with pytest.raises(ModelError, match="expanding window"):
        bad.build()
    assert not bad.check().ok and any("expanding window" in d.message for d in bad.check().diagnostics)
    assert "cumavg" not in ModelSource(text.format("0")).completions(6, 1)
    with pytest.raises(Exception, match="expanding window"):
        diff(parse_expression("cumavg(x)"), Name("x"))
    with pytest.raises(Exception, match="works on series"):  # not inside estimation: arrays
        from pymetrix.expr import evaluate

        evaluate(parse_expression("cumavg(x)"), lambda name, lag: np.arange(5.0))
    with pytest.raises(Exception, match=r"cumcov\(\) takes two series"):
        parse_expression("cumcov(x)")
    assert "centred window" in function_help("cmovavg") and "expanding window" in function_help("cumsd")
