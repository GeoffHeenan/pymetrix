"""EViews programs translated, phase 2 (estimation, tests and models): each translated estimate checked against an
independent computation written here (NumPy least squares, two-stage least squares, a Newton logit, the Wald and
White formulas, an ADF regression, a linear model's response worked out by hand), the bundle's own EViews estimates,
the report's statuses and reasons, and the script and the steps giving equal objects.
"""

import dataclasses
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Matrix
from pymetrix.commands import Session
from pymetrix_eviews import compare_estimates, translate_program

FIXTURES = Path(__file__).parent / "fixtures"
NAN = np.nan
N = 80
INDEX = pd.period_range("1990Q1", periods=N, freq="Q", name="time")


def _same(a, b):
    """The script's object and the steps' are equal (results by their numbers)."""
    if isinstance(a, Dataset):
        pd.testing.assert_frame_equal(a.frame, b.frame)
    elif isinstance(a, Matrix):
        np.testing.assert_array_equal(a.values, b.values)
    elif isinstance(a, float):
        assert (math.isnan(a) and math.isnan(b)) or a == b
    elif isinstance(a, dict):
        assert list(a) == list(b)
        for k in a:
            pd.testing.assert_series_equal(a[k], b[k])
    elif hasattr(a, "statistic"):
        assert type(a) is type(b) and str(a) == str(b)
    elif type(a).__name__ == "Template":  # a model read: its equations
        assert [str(e) for e in a.model.equations] == [str(e) for e in b.model.equations]
    elif hasattr(a, "equations") and hasattr(a, "solve"):  # a Model
        assert [str(e) for e in a.equations] == [str(e) for e in b.equations]
    elif hasattr(a, "summary"):
        assert type(a) is type(b) and str(a.summary()) == str(b.summary())
    elif hasattr(a, "names") and hasattr(a, "data"):  # a Selection
        assert list(a.names) == list(b.names)
    elif dataclasses.is_dataclass(a):  # a result without a summary (Johansen's): field by field
        assert type(a) is type(b)
        for f in dataclasses.fields(a):
            x, y = getattr(a, f.name), getattr(b, f.name)
            if isinstance(x, (np.ndarray, pd.DataFrame, pd.Series)):
                np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
            elif not callable(x) and f.compare and f.name != "provenance":
                assert str(x) == str(y)
    else:
        assert a == b


def _both(t, session=None):
    namespace: dict = {}
    exec(compile(t.script, "<script>", "exec"), namespace)
    out = t.run(session)
    s = out["session"]
    for name, value in {**{k: v for k, v in out.items() if k != "session"}, **s.objects}.items():
        assert name in namespace, f"{name} is made by the steps but not by the script"
        _same(value, namespace[name])
    return namespace, out


@pytest.fixture(scope="module")
def data_file(tmp_path_factory):
    """Quarterly data, 1990Q1 to 2009Q4, from a seeded generator: y on x and z, w an instrument for x, d a 0/1
    variable, l1 and l2 random walks."""
    rng = np.random.default_rng(20261124)
    x = rng.standard_normal(N)
    z = rng.standard_normal(N)
    u = rng.standard_normal(N)
    w = x + 0.6 * rng.standard_normal(N)
    y = 1.0 + 2.0 * x - 0.5 * z + u + 0.4 * (x - w)  # x correlated with the error through w's noise
    d = (0.3 + x + rng.standard_normal(N) > 0).astype(float)
    y2 = 0.5 + 0.3 * x + rng.standard_normal(N)
    l1 = np.cumsum(rng.standard_normal(N))
    frame = pd.DataFrame({"time": INDEX.astype(str), "y": y, "x": x, "z": z, "w": w, "d": d, "y2": y2, "l1": l1,
                          "l2": l1 + rng.standard_normal(N)})
    path = tmp_path_factory.mktemp("estimation") / "est.csv"
    frame.to_csv(path, index=False)
    return path


def _translate(program, path):
    return translate_program(program, bundle=path, read_options={"time": "time", "freq": "Q"})


def _frame(path):
    f = pd.read_csv(path)
    f.index = pd.PeriodIndex(f.pop("time"), freq="Q")
    return f


def _ols(y, X):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    n, k = X.shape
    s2 = e @ e / (n - k)
    XtXi = np.linalg.inv(X.T @ X)
    return b, e, s2, XtXi


# -- equations: least squares, its standard errors, the values read from it -----------------------------------------


PROGRAM = """wfopen est
smpl 1991q1 2007q4
equation eq1.ls y c x z
equation eq2.ls(cov=white) y c x z
equation eqx.ls log(y2+10) = c(1) + c(2)*x
eq1.wald c(2)=2, c(3)=0
scalar b2 = eq1.@coefs(2)
scalar se3 = eq1.@stderrs(3)
scalar t2 = eq1.@tstats(2)
scalar r2 = eq1.@r2
scalar aic = eq1.@aic
scalar dw = eq1.@dw
scalar nobs = eq1.@regobs
scalar k = eq1.@ncoef
scalar last = c(2)
eq1.makeresids r1
equation iv.tsls y c x z @ c w z
iv.testdrop z
smpl 2008q1 2009q4
eq1.fit yf yf_se
eq1.forecast yd
smpl @all
"""


def test_least_squares_and_what_is_read_from_it(data_file):
    t = _translate(PROGRAM, data_file)
    assert t.report.counts() == {"translated": 14, "translated with a note": 8, "left out": 0, "error": 0}
    _, out = _both(t)
    s = out["session"]
    f = _frame(data_file)
    est = f.loc["1991Q1":"2007Q4"]
    X = np.column_stack([np.ones(len(est)), est["x"], est["z"]])
    b, e, s2, XtXi = _ols(est["y"].to_numpy(), X)
    se = np.sqrt(s2 * np.diag(XtXi))
    np.testing.assert_allclose(s["eq1"].params.values, b, rtol=1e-12)
    np.testing.assert_allclose(s["eq1"].params.se, se, rtol=1e-10)
    # White's (HC1) standard errors: (X'X)^-1 X' diag(e^2) X (X'X)^-1 * n / (n - k)
    n, k = X.shape
    hc1 = XtXi @ (X.T * e**2) @ X @ XtXi * n / (n - k)
    np.testing.assert_allclose(s["eq2"].params.values, b, rtol=1e-12)
    np.testing.assert_allclose(s["eq2"].params.se, np.sqrt(np.diag(hc1)), rtol=1e-10)
    # an expression specification: c(1), c(2) are the parameters c_1, c_2, least squares on log(y2 + 10)
    bx, *_ = _ols(np.log(est["y2"].to_numpy() + 10), X[:, :2])
    np.testing.assert_allclose(s["eqx"].params.values, bx, rtol=1e-8)
    assert list(s["eqx"].params.names) == ["c_1", "c_2"]
    # the Wald test of c(2) = 2 and c(3) = 0: (Rb - r)' (R V R')^-1 (Rb - r)
    R = np.array([[0, 1.0, 0], [0, 0, 1.0]])
    gap = R @ b - np.array([2.0, 0.0])
    chi2 = gap @ np.linalg.solve(R @ (s2 * XtXi) @ R.T, gap)
    assert float(s["eq1_wald"].statistic) == pytest.approx(chi2, rel=1e-10)
    # the values read from the result, as EViews's data members give them
    assert out["b2"] == pytest.approx(b[1], rel=1e-12) and out["se3"] == pytest.approx(se[2], rel=1e-10)
    assert out["t2"] == pytest.approx(b[1] / se[1], rel=1e-10)
    r2 = 1 - e @ e / np.sum((est["y"] - est["y"].mean()) ** 2)
    assert out["r2"] == pytest.approx(r2, rel=1e-12)
    loglike = -n / 2 * (1 + math.log(2 * math.pi) + math.log(e @ e / n))
    assert out["aic"] == pytest.approx((-2 * loglike + 2 * k) / n, rel=1e-12)  # per observation, as EViews
    assert out["dw"] == pytest.approx(np.sum(np.diff(e) ** 2) / (e @ e), rel=1e-12)
    assert (out["nobs"], out["k"]) == (68.0, 3.0)
    assert out["last"] == pytest.approx(bx[1], rel=1e-8)  # c(2): the last estimation that fills c is eqx's
    # makeresids: the residuals over the estimation sample, missing elsewhere
    r1 = s["est"].frame["r1"]
    np.testing.assert_allclose(r1.loc["1991Q1":"2007Q4"], e, rtol=1e-9, atol=1e-12)
    assert r1.loc[:"1990Q4"].isna().all() and r1.loc["2008Q1":].isna().all()
    # two-stage least squares: x instrumented by w, z exogenous
    Z = np.column_stack([np.ones(len(est)), est["w"], est["z"]])
    Xhat = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    biv = np.linalg.solve(Xhat.T @ X, Xhat.T @ est["y"].to_numpy())
    np.testing.assert_allclose(s["iv"].params.values, biv, rtol=1e-10)
    # testdrop z: the Wald test of z's coefficient
    assert "z = 0" in t.report.line(18).python
    # fit (static) and forecast (dynamic, no lags: the same) over 2008Q1 to 2009Q4: b'x, missing elsewhere
    tail = f.loc["2008Q1":"2009Q4"]
    want = b[0] + b[1] * tail["x"] + b[2] * tail["z"]
    page = s["est"].frame
    np.testing.assert_allclose(page["yf"].loc["2008Q1":], want, rtol=1e-12)
    np.testing.assert_allclose(page["yd"].loc["2008Q1":], want, rtol=1e-12)
    assert page["yf"].loc[:"2007Q4"].isna().all()
    assert (page["yf_se"].loc["2008Q1":] > math.sqrt(s2)).all()  # the forecast's standard error exceeds the regression's


def test_c_holds_the_last_estimation_that_uses_it(data_file):
    t = _translate("wfopen est\nequation a.ls y c x\nequation b.ls y = c(1) + c(2)*z\nscalar s = c(2)\n", data_file)
    assert t.report.line(4).python == 's = float(b.params["c_2"])'
    assert "the last estimation's coefficients, b's" in t.report.line(4).reason


def test_the_report_and_the_script(data_file):
    t = _translate(PROGRAM, data_file)
    rows = {r.line: r for r in t.report}
    assert rows[3].python == 'eq1 = Equation("y ~ x + z", name="eq1").fit(est)'
    assert rows[4].python == 'eq2 = Equation("y ~ x + z", name="eq2").fit(est, method=OLS(cov="white"))'
    assert "HC1" in rows[4].reason
    assert rows[5].python == 'eqx = Equation("log(y2 + 10) ~ c_1 + c_2*x", params=("c_1", "c_2"), name="eqx").fit(est)'
    assert rows[6].python == 'eq1_wald = eq1.test("wald", "x=2, z=0")'
    assert rows[7].python == 'b2 = float(eq1.params["x"])'
    assert rows[11].reason.startswith("@aic per observation")
    assert rows[16].python == "est = est.with_series(r1=eq1.resid)"
    assert rows[17].python == ('iv = Equation("y ~ x + z", name="iv").fit(est, method=IV(endog=("x",), instruments=("w",)))')
    assert rows[20].python.splitlines() == ['eq1_fit = eq1.forecast(est, ("2008Q1", "2009Q4"), dynamic=False)',
                                            "est = est.with_series(yf=eq1_fit['forecast'])",
                                            "est = est.with_series(yf_se=eq1_fit['se'])"]
    assert "from pymetrix import Dataset, Sample, Equation, IV, OLS" in t.script


def test_steps_record_calls_that_replay(data_file):
    t = _translate(PROGRAM, data_file)
    s = Session()
    t.run(s)
    codes = [r.code for r in s.history]
    assert "eq1 = Equation('y ~ x + z', name='eq1').fit(est)" in codes
    assert "est = est.with_series(r1=eq1.resid)" in codes
    assert all(r.ok for r in s.history)
    ns: dict = {}
    exec(s.script(), ns)
    np.testing.assert_array_equal(ns["eq2"].params.se, s["eq2"].params.se)
    pd.testing.assert_frame_equal(ns["est"].frame, s["est"].frame)


# -- other estimators ----------------------------------------------------------------------------------------------


def test_logit_against_newton_and_estimators_with_notes(data_file):
    t = _translate("wfopen est\nequation lg.binary(d=l) d c x\nequation pr.binary d c x\nls y c x ar(1)\n"
                   "ls(cov=hac) y c x\nequation ga.arch(1,1) y c x\nequation gm.gmm y c x z @ c w z\n"
                   "equation q.qreg(quant=0.25) y c x\nequation ar.ardl(deplags=2, reglags=1, fixed) y x\n", data_file)
    _, out = _both(t)
    s = out["session"]
    f = _frame(data_file)
    X = np.column_stack([np.ones(N), f["x"]])
    dd = f["d"].to_numpy()
    b = np.zeros(2)
    for _ in range(30):  # Newton's method on the logit log likelihood
        p = 1 / (1 + np.exp(-X @ b))
        b = b + np.linalg.solve((X.T * (p * (1 - p))) @ X, X.T @ (dd - p))
    np.testing.assert_allclose(s["lg"].params.values, b, rtol=1e-8)
    rows = {r.line: r for r in t.report}
    assert "Probit()" in rows[3].python and "EViews's default" in rows[3].reason
    assert rows[4].status == "translated with a note" and "ARMA" in rows[4].reason and "errors=ARMA(ar=1)" in rows[4].python
    assert "bandwidth" in rows[5].reason
    assert "variance=GARCH(arch=1, garch=1)" in rows[6].python and "backcasts" in rows[6].reason
    assert 'GMM(instruments=("w", "z"))' in rows[7].python
    assert "Quantile(q=0.25)" in rows[8].python
    assert rows[9].python == 'ar = ARDL("y ~ x", lags=2, order=1).fit(est)'
    # quantile regression's estimate: the 0.25 quantile's check function at its minimum, a vertex through two points
    q = s["q"].params.values
    loss = lambda c: np.sum(np.maximum(0.25 * (f["y"] - X @ c), -0.75 * (f["y"] - X @ c)))  # noqa: E731
    assert all(loss(q) <= loss(q + step) + 1e-9 for step in ([1e-4, 0], [-1e-4, 0], [0, 1e-4], [0, -1e-4]))


def test_var_vec_and_a_system_against_least_squares(data_file):
    t = _translate("wfopen est\nvar v.ls 1 2 y x @ c\nvar ve.ec(c, 1) 1 1 l1 l2\nsystem sy\nsy.append y = c(1) + c(2)*x\n"
                   "sy.append y2 = c(3) + c(4)*x\nsy.sur\nscalar b4 = sy.@coefs(4)\n", data_file)
    _, out = _both(t)
    s = out["session"]
    f = _frame(data_file)
    # the VAR's equations: each by least squares on a constant and two lags of y and x
    Y = f[["y", "x"]].to_numpy()
    X = np.column_stack([np.ones(N - 2), Y[1:-1], Y[:-2]])
    B = np.linalg.lstsq(X, Y[2:], rcond=None)[0]
    fitted = s["v"].fitted.frame[["y", "x"]].dropna().to_numpy()
    np.testing.assert_allclose(fitted, X @ B, rtol=1e-9, atol=1e-12)
    assert "VECM((\"l1\", \"l2\"), lags=1, rank=1)" in t.report.line(3).python
    # SUR with the same regressors in each equation is least squares equation by equation
    Xs = np.column_stack([np.ones(N), f["x"]])
    b_y = np.linalg.lstsq(Xs, f["y"], rcond=None)[0]
    b_y2 = np.linalg.lstsq(Xs, f["y2"], rcond=None)[0]
    np.testing.assert_allclose(s["sy"].params.values, np.r_[b_y, b_y2], rtol=1e-8)
    assert out["b4"] == pytest.approx(b_y2[1], rel=1e-8)


def test_tests_on_an_equation_and_on_series(data_file):
    t = _translate("wfopen est\nequation eq.ls y c x z\neq.auto(2)\neq.hettest(type=white)\neq.archtest(1)\neq.reset(1)\n"
                   "eq.chow 2000q1\neq.hist\neq.rls(q)\ny.uroot(adf, const, lag=1)\ngroup g l1 l2\ng.coint(c, 2)\n", data_file)
    _, out = _both(t)
    s = out["session"]
    rows = {r.line: r for r in t.report}
    assert rows[3].python == 'eq_auto = eq.test("breusch_godfrey", lags=2)'
    assert rows[4].python == 'eq_hettest = eq.test("white")'
    assert rows[6].python == 'eq_reset = eq.test("reset", power=2)'
    assert rows[7].python == 'eq_chow = eq.test("chow", at="2000Q1")'
    assert rows[8].python == 'eq_hist = eq.test("jarque_bera")'
    assert rows[9].python == "eq_rls = eq.recursive_residuals()"
    assert rows[12].python == 'g_coint = est.johansen(("l1", "l2"), lags=2)'
    # the ADF statistic with one lagged difference and a constant: the t statistic on y(-1) in
    # d(y) = a + rho y(-1) + g d(y(-1))
    y = _frame(data_file)["y"].to_numpy()
    dy = np.diff(y)
    X = np.column_stack([np.ones(N - 2), y[1:-1], dy[:-1]])
    b, e, s2, XtXi = _ols(dy[1:], X)
    assert float(s["y_uroot"].statistic) == pytest.approx(b[1] / math.sqrt(s2 * XtXi[1, 1]), rel=1e-9)


# -- what is left out, and why --------------------------------------------------------------------------------------


@pytest.mark.parametrize("line, words", [
    ("equation e.binary(d=x) d c x", ("extreme-value", "gap")),
    ("eq.testadd w", ("testadd", "gap")),
    ("ls(w=x) y c x", ("option w", "would differ")),
    ("equation e.stepls y c x", ("stepwise",)), ("breakls y c x", ("bai_perron",)), ("threshold y c x", ("ThresholdAR",)),
    ("switchreg y c x", ("MarkovSwitching",)), ("glm y c x", ("family",)), ("nls y = c(1)*x", ("explicit form",)),
    ("equation e.ls y c x sar(4)", ("seasonal ARMA",)), ("equation e.ls y c pdl(x, 4, 2)", ("polynomial",)),
    ("equation e.ordered(d=x) d x", ("extreme-value",)), ("equation e.count(d=e) d c x", ("Poisson",)),
    ("equation e.censored(d=l) y c x", ("Tobit",)), ("var v.ls 1 2 4 4 y x", ("lags",)),
    ("series s = c(1)*x", ("scalar b = c(1)",)), ("eq.auto", ("order",)), ("m.solve", ("not an object",)),
    ("series r = resid", ("makeresids",)),
])
def test_lines_left_out_give_a_reason(data_file, line, words):
    t = _translate(f"wfopen est\nequation eq.ls y c x\n{line}\n", data_file)
    row = t.report.line(3)
    assert row.status in ("left out", "error"), (row.status, row.reason)
    for w in words:
        assert w in row.reason, row.reason
    assert t.report.counts()["error"] == (1 if line == "m.solve" else 0)


# -- the bundle's equations: EViews's own estimates ---------------------------------------------------------------


def test_bundle_equations_reestimated_match_eviews(tmp_path):
    shutil.copytree(FIXTURES / "macro", tmp_path / "macro")
    bundle = tmp_path / "macro" / "macro.evbundle"
    t = translate_program("wfopen macro\nsmpl 1980q1 1982q4\nequation eq_cons.ls cons c inc\nequation eq_rev.ls cons inc c\n"
                          "equation eq_white.ls(cov=white) cons c inc\nscalar b = eq_rev.@coefs(1)\n", bundle=bundle)
    with pytest.warns(Warning):  # the bundle reader notes the names it changed in the cross page
        _, out = _both(t)
    s = out["session"]
    for name, order in (("eq_cons", [0, 1]), ("eq_rev", [1, 0]), ("eq_white", [0, 1])):
        rows = (tmp_path / "macro" / "objects" / "quarterly" / f"{name}_est.csv").read_text().splitlines()[1:]
        eviews = np.array([[float(v) for v in r.split(",")[1:]] for r in rows])  # EViews's c(1), c(2): estimate, s.e.
        fit = s[name]
        # pymetrix's const first: EViews's c(i) reordered to it
        np.testing.assert_allclose(fit.params.values, eviews[order, 0], rtol=1e-9)
        if name != "eq_white":  # the fixture's White standard errors are placeholders, not EViews's numbers
            np.testing.assert_allclose(fit.params.se, eviews[order, 1], rtol=1e-9)
    assert out["b"] == pytest.approx(0.938413540386753, rel=1e-9)  # eq_rev's c(1) is inc's coefficient
    table = compare_estimates(Dataset.read(str(bundle), page="quarterly")).to_frame()
    assert list(table["result"][:4]) == ["matches"] * 4  # the plugin's own check agrees


# -- models ----------------------------------------------------------------------------------------------------------


MODEL = """wfopen est
smpl 1990q2 2009q4
model m
m.append ys = 0.5*ys(-1) + x
m.append @identity tot = ys + z
smpl 2000q1 2003q4
m.addassign @all
m.addinit(v=n) @all
m.solve
m.scenario(n, a=_1) "shock"
series x_1 = x
smpl 2000q1 2000q1
x_1 = x + 1
smpl 2000q1 2003q4
m.override x
m.solveopt(o=g, m=500)
m.solve
m.scenario(n, a=_2) "held"
m.exclude ys
m.solve
"""


def _model_data(data_file, tmp_path):
    f = _frame(data_file)
    ys = np.zeros(N)
    for i in range(1, N):
        ys[i] = 0.5 * ys[i - 1] + f["x"].iloc[i] + 0.1 * f["z"].iloc[i]  # the data are not the model's own path
    f["ys"] = ys
    f["tot"] = ys + f["z"]
    out = tmp_path / "est.csv"
    frame = f.copy()
    frame.insert(0, "time", f.index.astype(str))
    frame.to_csv(out, index=False)
    return out, f


def test_a_model_tracked_shocked_and_held(data_file, tmp_path):
    path, f = _model_data(data_file, tmp_path)
    t = _translate(MODEL, path)
    assert t.report.counts()["error"] == 0 and t.report.counts()["left out"] == 0
    _, out = _both(t)
    page = out["session"]["est"].frame
    span = slice("2000Q1", "2003Q4")
    # addinit: the add factors make the model reproduce the data in the baseline
    np.testing.assert_allclose(page["ys_0"].loc[span], f["ys"].loc[span], rtol=1e-10)
    np.testing.assert_allclose(page["tot_0"].loc[span], f["tot"].loc[span], rtol=1e-10)
    # x one higher in 2000Q1 only: ys moves by 1, 0.5, 0.25, ... (the add factors cancel in the difference)
    response = (page["ys_1"] - page["ys_0"]).loc[span].to_numpy()
    np.testing.assert_allclose(response, 0.5 ** np.arange(16), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose((page["tot_1"] - page["tot_0"]).loc[span], 0.5 ** np.arange(16), rtol=1e-9, atol=1e-12)
    # ys excluded: held at its data, so tot = ys + z there
    np.testing.assert_allclose(page["ys_2"].loc[span], f["ys"].loc[span], rtol=1e-12)
    rows = {r.line: r for r in t.report}
    assert "m = read_eviews(" in rows[8].python and "m_model = m.model" in rows[8].python
    assert 'm_addfactors = m_model.add_factors(est, span=("2000Q1", "2003Q4"))' in rows[8].python
    assert 'scenario=Scenario(adjust=m_addfactors, name="shock")' in rows[17].python
    assert "solver=GaussSeidel(max_iter=500)" in rows[17].python
    assert 'est_1 = est.with_series(x="x_1")' in rows[17].python
    assert 'fix={"ys": None}' in rows[20].python


def test_a_model_read_with_a_merged_equation(data_file):
    t = _translate("wfopen est\nsmpl 1991q1 2007q4\nequation eq.ls y c x z\nmodel m\nm.merge eq\nm.append @identity w2 = y + w\n"
                   "smpl 2008q1 2009q4\nm.solve\n", data_file)
    _, out = _both(t)
    s = out["session"]
    f = _frame(data_file)
    b = s["eq"].params.values
    want = b[0] + b[1] * f["x"] + b[2] * f["z"]
    np.testing.assert_allclose(s["est"].frame["y_0"].loc["2008Q1":], want.loc["2008Q1":], rtol=1e-10)
    np.testing.assert_allclose(s["est"].frame["w2_0"].loc["2008Q1":], (want + f["w"]).loc["2008Q1":], rtol=1e-10)
    assert "data=est" in t.report.line(8).python


def test_the_model_readers_refusals_at_their_lines():
    t = translate_program("model m\nscalar f\nm.append y = f*x\nm.append z = y + 1\n")
    rows = {r.line: r for r in t.report}
    assert rows[2].status == "translated with a note" and "the model reader: the scalar f is declared without a value" in rows[2].reason
    for line in (1, 3, 4):
        assert rows[line].status == "left out" and rows[line].reason.startswith("the model m is not read: line 2:")
    t = translate_program("model m\nscalar f = 0.5\nm.append y = f*x\nm.append z = y + 1\n")
    assert t.report.counts() == {"translated": 1, "translated with a note": 3, "left out": 0, "error": 0}
    _, out = _both(t)
    m = out["session"]["m"]
    assert [str(e) for e in m.model.equations] == ["Equation('y ~ f*x', params=['f'])", "Equation('z = y + 1')"]
    assert m.model.params == {"f": 0.5}
    assert 'params={"f": f}' in t.report.line(4).python  # read after its last line, nothing using it later


def test_models_made_by_lines_left_out_say_so():
    t = translate_program("wfcreate q 2000q1 2001q4\n%m = \"mod\"\ncall make(%m)\n{%m}.solve\n")
    assert t.report.line(4).status == "left out" and "mod is not made by a line translated so far" in t.report.line(4).reason


def test_solver_options_and_model_lines_the_reader_refuses():
    t = translate_program("model m\nm.append y = 0.5*y(-1) + x\nm.append @add q\nm.append @innov y 0.1\n"
                          "m.solveopt(o=b, g=12, z=1e-12)\nm.solveopt(m=abc)\n")
    rows = {r.line: r for r in t.report}
    # @add of a variable no equation determines: the reader's message at the model's lines, naming line 3
    assert rows[3].status == "left out" and "line 3: no equation determines q" in rows[3].reason
    assert rows[1].reason == rows[3].reason
    t = translate_program("wfcreate q 2000q1 2001q4\nseries x = 1\nseries y = 0\nmodel m\nm.append y = 0.5*y(-1) + x\n"
                          "m.append @innov y 0.1\nm.solveopt(o=b, g=12, z=1e-12)\nsmpl 2000q2 2001q4\nm.solve\nm.solveopt(m=abc)\n")
    rows = {r.line: r for r in t.report}
    assert "o=b (Broyden): pymetrix's Newton" in rows[7].reason and "not mapped: g=12, z=1e-12" in rows[7].reason
    assert "the innovations (y_innov) are zero" in rows[9].reason
    _, out = _both(t)
    # y = 0.5 y(-1) + 1 from y = 0 in 2000Q1: 1, 1.5, 1.75, ...
    np.testing.assert_allclose(out["session"]["untitled"].frame["y_0"].iloc[1:], 2 - 0.5 ** np.arange(7), rtol=1e-12)
    assert rows[10].status == "left out" and "m=abc is not translated: a number is" in rows[10].reason
