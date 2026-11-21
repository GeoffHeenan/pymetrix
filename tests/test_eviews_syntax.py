"""EViews syntax in the model editor (``pymetrix.models.eviews``): model text and a program subset read into templates.

Checked against what:
- a small macro model in EViews's model text (consumption and investment with c(i) coefficients, @identity for GDP
  and income, lags, a dlog equation, add factors) builds exactly the Model written by hand in pymetrix code (the same
  equations' residuals, parameters and roles), and its Newton solution path equals the hand-written model's (1e-12);
- its program form (``model``, ``.merge``, ``.append``, ``equation eq.ls`` list specifications, ``smpl``) fitted to a
  dataset gives the coefficients of the pymetrix Equations fitted directly (1e-10); without data, a specification
  takes the values given to c(i), as EViews's coefficient vector holds them;
- every function read is evaluated on data against the pymetrix expression (or the arithmetic) it stands for (1e-12),
  and the functions refused are refused at their line and column;
- writing: EViews text -> build -> EViews -> build gives the same solution path; pymetrix -> EViews -> pymetrix keeps
  the template's steady state and responses (the Translation check, and again by hand); a backward-looking model
  written in Dynare's syntax says it has no shocks; what does not carry across is said in notes;
- planted problems (a coefficient without a value, an unsupported function, two equations for a variable, an unclosed
  bracket, an add factor for no equation, an unknown linked equation with a suggestion, the programming language, an
  estimation method other than least squares, -a^b) are each reported once at their line and column;
- the FRB/US excerpt (tests/fixtures/eviews: six variables of the Federal Reserve's public FRB/US package's
  mods/model.xml, and their series in LONGBASE.TXT, 2025Q1-2027Q4) reads, each equation's residual equals that of the
  Fed's own Python twin of the equation (1e-10, on random data), and add factors computed from the baseline make
  the model reproduce it when solved; with PYMETRIX_FRBUS (the frbus_package folder) every equation of model.xml is
  checked the same way and the whole model compiled, and with PYMETRIX_FRBUS_DATA too (the data_only_package folder)
  the full model's solution over 40 quarters of projection reproduces LONGBASE (1e-6 relative);
- the editor (offscreen Qt): the EViews syntax in the combo, its template checks clean, highlighting runs, Write as
  EViews gives a checked translation;
- J.-L. Brillet's book of EViews models (opt-in: the book is copyright and none of it is in the repository). Set
  PYMETRIX_BRILLET to a folder holding your own copies of its files: brillet_basic.prg (the very basic model of
  chapter 5), brillet_small_model.txt (the estimated small model of section 6.3.7) and structmodel.txt (the book's
  text, for the larger model of chapter 11); PYMETRIX_BRILLET may also name structmodel.txt itself. Each test skips
  when its file is not there.
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, ModelSource, Newton
from pymetrix.metadata import Role
from pymetrix.models import Model, templates
from pymetrix.models.eviews import convert_expression, read_eviews, read_eviews_xml, xml_equation
from pymetrix.models.solvers import ModelError
from pymetrix.models.writers import export_model

FIXTURES = Path(__file__).parent / "fixtures" / "eviews"

TEXT = """' a small macro model in EViews's model text
cons = c(1) + c(2)*inc + c(3)*cons(-1)
dlog(inv) = c(4) + c(5)*dlog(gdp(-1)) + c(6)*log(inv(-1)/gdp(-1))
@identity gdp = cons + inv + gov
@identity inc = gdp - tax
@add cons cons_a
@add inv inv_a
' the coefficients
c(1) = 10
c(2) = 0.6
c(3) = 0.2
param c(4) 0.01 c(5) 0.3 c(6) -0.05
"""
VALUES = {"c_1": 10.0, "c_2": 0.6, "c_3": 0.2, "c_4": 0.01, "c_5": 0.3, "c_6": -0.05}


def _by_hand() -> Model:
    return Model([Equation("cons ~ c_1 + c_2*inc + c_3*cons[-1] + cons_a", params=["c_1", "c_2", "c_3"], name="cons"),
                  Equation("dlog(inv) ~ c_4 + c_5*dlog(gdp[-1]) + c_6*log(inv[-1]/gdp[-1]) + inv_a", params=["c_4", "c_5", "c_6"],
                           name="inv"),
                  Equation("gdp = cons + inv + gov"), Equation("inc = gdp - tax")], params=VALUES)


def _data(n: int = 52, seed: int = 0) -> Dataset:
    """The model's path with noisy add factors (the data a program's specifications are fitted to)."""
    rng = np.random.default_rng(seed)
    periods = pd.period_range("2000Q1", periods=n, freq="Q", name="time")
    t = np.arange(n, dtype=float)
    frame = pd.DataFrame({"gov": 20 + 0.5 * t, "tax": 10 + 0.2 * t, "cons_a": rng.normal(0, 1.0, n), "inv_a": rng.normal(0, 0.02, n),
                          "cons": np.nan, "inv": np.nan, "gdp": np.nan, "inc": np.nan}, index=periods)
    frame.iloc[:2, frame.columns.get_indexer(["cons", "inv", "gdp", "inc"])] = [[60.0, 15.0, 95.0, 85.0], [61.0, 15.5, 96.5, 86.4]]
    solved = _by_hand().solve(Dataset.from_frame(frame), (str(periods[2]), str(periods[-1])), solver=Newton())
    for v in ("cons", "inv", "gdp", "inc"):
        frame[v] = solved.frame[v].to_numpy()
    return Dataset.from_frame(frame)


def _blank(data: Dataset, start: str) -> Dataset:
    frame = data.frame.copy()
    frame.loc[pd.Period(start, "Q"):, ["cons", "inv", "gdp", "inc"]] = np.nan
    return Dataset.from_frame(frame)


def _residuals(model: Model) -> list[str]:
    return [f"{e.name}: {e.dependent}: {e.residual}" for e in model.model_equations]


def test_model_text_builds_the_hand_written_model():
    found = ModelSource(TEXT, "eviews").check()
    assert found.ok and not found.diagnostics, found.diagnostics
    assert found.role("endogenous") == ("cons", "inv", "gdp", "inc")
    assert set(found.role("exogenous")) == {"gov", "tax", "cons_a", "inv_a"}
    assert dict((s.name, s.value) for s in found.symbols if s.role == "parameter") == VALUES
    built = ModelSource(TEXT, "eviews").build()
    hand = _by_hand()
    assert _residuals(built.model) == _residuals(hand)
    assert built.model.endogenous == hand.endogenous and built.model.exogenous == hand.exogenous
    assert built.model.parameters.values() == hand.parameters.values()
    assert built.shocks == () and built.model.compile().roles == hand.compile().roles
    data = _blank(_data(), "2003Q1")
    span = ("2003Q1", "2012Q4")
    a = built.model.solve(data, span, solver=Newton()).frame
    b = hand.solve(data, span, solver=Newton()).frame
    for v in ("cons", "inv", "gdp", "inc"):
        assert np.allclose(a[v], b[v], rtol=0, atol=1e-12)
    assert np.allclose(a["cons"].loc[pd.Period("2003Q1", "Q"):], _data().frame["cons"].loc[pd.Period("2003Q1", "Q"):], atol=1e-9)


def test_names_in_any_case_and_innovations():
    upper = TEXT.replace("cons", "CONS").replace("c(1)", "C(1)").replace("@identity", "@IDENTITY") + "@innov inv 0.02\n"
    built = read_eviews(upper)
    assert built.model.endogenous == ("cons", "inv", "gdp", "inc")
    assert built.shocks == ("inv_innov",) and built.shock_sd == {"inv_innov": 0.02}
    assert built.model.compile().roles["inv_innov"] == Role.SHOCK
    assert str(built.model.model_equations[1].residual).endswith("+ inv_a + inv_innov)")
    assert read_eviews(TEXT, params={"c(1)": 12.0}).model.parameters.values()["c_1"] == 12.0


PROGRAM = """' the same model as a program, its behavioural equations estimated
wfcreate q 2000q1 2012q4
smpl 2001q1 2012q4
equation eqc.ls cons c inc cons(-1)
equation eqi.ls(cov=hac) dlog(inv) c dlog(gdp(-1)) log(inv(-1)/gdp(-1))
model macro
macro.merge eqc
macro.merge eqi
macro.append @identity gdp = cons + inv + gov
macro.append @identity inc = gdp - tax
macro.solve
"""


def test_program_specifications_fitted_as_pymetrix_equations():
    data = _data()
    found = ModelSource(PROGRAM, "eviews").check()
    assert found.ok
    notes = [(d.line, d.column, d.severity) for d in found.diagnostics]
    assert (2, 1, "note") in notes and (11, 1, "note") in notes and (5, 17, "note") in notes  # ignored: wfcreate, solve; options
    assert [d.severity for d in found.diagnostics if "is estimated" in d.message] == ["warning", "warning"]
    with pytest.raises(ModelError, match="is estimated"):
        ModelSource(PROGRAM, "eviews").build()
    built = read_eviews(PROGRAM, data=data)
    assert built.name == "macro"
    span = ("2001Q1", "2012Q4")
    direct = [Equation("cons ~ inc + cons[-1]").fit(data, span=span),
              Equation("dlog(inv) ~ dlog(gdp[-1]) + log(inv[-1]/gdp[-1])").fit(data, span=span)]
    for fitted, reference in zip(built.model.equations[:2], direct):
        assert np.allclose(np.asarray(fitted.params.values, float), np.asarray(reference.params.values, float), rtol=0, atol=1e-10)
        assert len(fitted.resid) == len(reference.resid) == 48
    assert built.model.equations[0].equation.name == "eqc"
    # without data, the list specification is c(1) + c(2)*inc + c(3)*cons(-1), with the values given to c
    given = PROGRAM.replace("equation eqi.ls(cov=hac) dlog(inv) c dlog(gdp(-1)) log(inv(-1)/gdp(-1))\n", "") \
        .replace("macro.merge eqi\n", "macro.append dlog(inv) = c(4) + c(5)*dlog(gdp(-1)) + c(6)*log(inv(-1)/gdp(-1))\n") \
        + "c(1) = 10\nc(2) = 0.6\nc(3) = 0.2\nparam c(4) 0.01 c(5) 0.3 c(6) -0.05\n"
    calibrated = read_eviews(given)
    assert [str(e.residual) for e in calibrated.model.model_equations] == \
        [str(e.residual).replace(" + cons_a", "").replace(" + inv_a", "") for e in _by_hand().model_equations]
    assert calibrated.model.equations[0].name == "eqc"


FUNCTIONS = [  # EViews, and the pymetrix expression (or arithmetic) it means
    ("log(x) + @log(y)", "log(x) + log(y)"), ("exp(x) - @exp(y)", "exp(x) - exp(y)"), ("abs(x - y) + @abs(y - x)", "2*abs(x - y)"),
    ("sqr(x)", "x^0.5"), ("@sqrt(x)", "sqrt(x)"), ("@inv(x)", "1/x"),
    ("d(x)", "x - x[-1]"), ("d(x, 2)", "x - 2*x[-1] + x[-2]"), ("d(x, 0, 4)", "x - x[-4]"), ("d(x, 1, 4)", "x - x[-1] - x[-4] + x[-5]"),
    ("d(log(x), 0, 1)", "log(x) - log(x[-1])"), ("dlog(x)", "log(x) - log(x[-1])"), ("dlog(x, 1, 4)", "dlog(x) - dlog(x[-4])"),
    ("dlog(x, 2)", "log(x) - 2*log(x[-1]) + log(x[-2])"), ("@pch(x)", "x/x[-1] - 1"), ("@pch(x + y)", "(x + y)/(x[-1] + y[-1]) - 1"),
    ("100*@pch(x)", "pch(x)"), ("@movav(x, 3)", "(x + x[-1] + x[-2])/3"), ("@movsum(x(-1), 2)", "x[-1] + x[-2]"),
    ("@movmax(x, 2)", "max(x, x[-1])"), ("@movmin(x, 2)", "min(x, x[-1])"), ("x(1) + x(+1) + x(-1)", "2*x[+1] + x[-1]"),
    ("@recode(x > y, x, y)", "max(x, y)"), ("@recode((x) >= (y(-1)), x, y(-1))", "max(x, y[-1])"),
    ("@recode(x < y, x, y)", "min(x, y)"), ("@recode(x > y, y, x)", "min(x, y)"),
    ("@cnorm(x) + @dnorm(y) + @qnorm(0.3)", "normcdf(x) + normpdf(y) + norminv(0.3)"),
    ("@ctdist(x, 5) + @dtdist(y, 5) + @qtdist(0.3, 5)", "tcdf(x, 5) + tpdf(y, 5) + tinv(0.3, 5)"),
    ("@cchisq(x, 3) + @dchisq(y, 3) + @qchisq(0.3, 3)", "chi2cdf(x, 3) + chi2pdf(y, 3) + chi2inv(0.3, 3)"),
    ("@cfdist(x, 3, 7) + @dfdist(y, 3, 7) + @qfdist(0.3, 3, 7)", "fcdf(x, 3, 7) + fpdf(y, 3, 7) + finv(0.3, 3, 7)"),
    (".5*x^-2 + (x^2)^0.5 + -(x^2)", "0.5/x^2 + abs(x) - x^2"),
    ("@pchy(x)", "x/x[-4] - 1"), ("@pcy(x)", "100*(x/x[-4] - 1)"), ("@pcha(x)", "(x/x[-1])^4 - 1"),
    ("@pca(x)", "100*((x/x[-1])^4 - 1)"), ("@pc(x)", "pch(x)"),
    ("(x <= 1)*x + (x <> y) + (x = x)", "(x <= 1)*x + (x != y) + 1"), ("@recode(x > 1, x, 0)", "(x > 1)*x"),
    ("@recode(x = y, 1, 2) + @recode(y < 1, 3, x)", "2 + (y < 1)*3 + (y >= 1)*x"),
    ("@elem(@trend(60:1), \"2004q4\") + @elem(@trend(\"1978Q2\"), \"1979q1\")", "179 + 3"),
]


@pytest.mark.parametrize("eviews, reference", FUNCTIONS)
def test_functions_mean_what_they_do_in_eviews(eviews, reference):
    rng = np.random.default_rng(1)
    periods = pd.period_range("2000Q1", periods=30, freq="Q", name="time")
    data = Dataset.from_frame(pd.DataFrame({"x": rng.uniform(0.5, 2, 30), "y": rng.uniform(0.5, 2, 30)}, index=periods))
    converted = convert_expression(eviews, frequency="Q")
    a, b = data.eval(converted).to_numpy(), data.eval(reference).to_numpy()
    assert np.array_equal(np.isnan(a), np.isnan(b)) and np.isfinite(a).sum() >= 24
    assert np.allclose(a[np.isfinite(a)], b[np.isfinite(b)], rtol=1e-12, atol=1e-12), (converted, reference)


def test_moving_average_against_the_arithmetic():
    rng = np.random.default_rng(2)
    x = rng.normal(size=20)
    periods = pd.period_range("2000Q1", periods=20, freq="Q", name="time")
    data = Dataset.from_frame(pd.DataFrame({"x": x}, index=periods))
    got = data.eval(convert_expression("@movav(x, 4)")).to_numpy()
    assert np.allclose(got[3:], pd.Series(x).rolling(4).mean().to_numpy()[3:], atol=1e-14) and np.isnan(got[:3]).all()


REFUSED = [  # an equation using it, the column where it is refused, and what the message says
    ("y = @pchy(x)", 5, "frequency"), ("y = @pcha(x)", 5, "frequency"), ("y = @max(x)", 5, "sample's maximum"),
    ("y = @elem(x(-1), \"2000q1\")", 5, "a series or @trend"), ("y = @trend(15q1)", 12, "not a date"),
    ("y = @elem(x, \"2000:1\")", 14, "not a date"), ("y = @seas(x)", 11, "period of the year"),
    ("y = @movstdev(x, 4)", 5, "degrees of freedom"), ("y = @movvar(x, 4)", 5, "degrees of freedom"),
    ("y = @recode(x, 1, 2) + 1", 5, "a comparison"), ("y = @round(x)", 5, "not read here"),
    ("y = -x^2", 5, "-a^b"), ("y = x^2^3", 8, "a^b^c"), ("y = x(t)", 5, "neither a lag"), ("y = lg(x)", 5, "log"),
    ("y = x and z", 7, "logical"), ("y = x + {%z}", 9, "substitution"), ("y = x > 1", 7, "in brackets"),
    ("y = dlog(x, n)", 13, "whole number"), ("y = log(x, 2)", 5, "1 argument"),
]


@pytest.mark.parametrize("line, column, says", REFUSED)
def test_unsupported_refused_where_they_are(line, column, says):
    text = "' an equation\nc(1) = 1\n" + line + "\n"
    found = ModelSource(text, "eviews").check()
    errors = found.errors
    assert len(errors) == 1, errors
    assert (errors[0].line, errors[0].column) == (3, column) and says in errors[0].message, errors[0]


PLANTED = """' planted problems
cons = c(1) + c(2)*inc + c(9)*cons(-1)
inv = c(4) + @pchy(gdp)
@identity gdp = cons + inv + gov
@identity gdp = cons + inv
inc = (gdp - tax
c(1) = 10
c(2) = 0.6
c(4) = 1
@add incc
@innov gdp 0.1
@identity yd = c(2)*inc
"""


def test_planted_problems_each_once_where_they_are():
    found = ModelSource(PLANTED, "eviews").check()
    got = [(d.line, d.column, d.message.split(":")[0].split(";")[0]) for d in found.diagnostics if d.severity == "error"]
    assert got == [(2, 26, "c(9) has no value"), (3, 14, "@pchy needs the data's frequency"),
                   (5, 11, "gdp is already determined by the equation on line 4"), (6, 7, "this bracket is never closed"),
                   (10, 6, "no equation determines incc"), (11, 8, "the equation for gdp is an identity"),
                   (12, 16, "an identity has no coefficients")]
    with pytest.raises(ModelError, match="line 2, column 26"):
        ModelSource(PLANTED, "eviews").build()


PROGRAM_PROBLEMS = """model m
for !i = 1 to 3
m.append invest = c(1)*invest(-1) + x
m.merge eqx
equation eqc.ls cons c inc
equation eqd.tsls cons c inc
n.append z = 1
include library
m.append x(-1) = 2
m.append w = invset + 1
c(1) = 0.5
coef(2) b
m.append v = b(1)*invest + b(3)
"""


def test_program_problems_each_once_where_they_are():
    found = ModelSource(PROGRAM_PROBLEMS, "eviews").check()
    got = [(d.line, d.column, d.severity, d.message) for d in found.diagnostics if d.severity != "note"]
    assert [(line, column, severity) for line, column, severity, _ in got] == [
        (2, 5, "error"), (4, 9, "error"), (6, 14, "error"), (7, 1, "error"), (8, 1, "error"), (9, 10, "error"), (10, 14, "warning"),
        (13, 14, "error")]
    messages = [m for *_, m in got]
    assert "programming language" in messages[0] and "did you mean 'eqc'?" in messages[1] and "least squares" in messages[2]
    assert "is not the model (m)" in messages[3] and "include" in messages[4] and "no series of the current period" in messages[5]
    assert "did you mean 'invest'?" in messages[6] and "b(1) has no value" in messages[7]


def test_symbols_completions_and_hover():
    source = ModelSource(TEXT, "eviews")
    found = source.check()
    roles = {s.name: s.role for s in found.symbols}
    assert roles["cons"] == "endogenous" and roles["gov"] == "exogenous" and roles["c_1"] == "parameter"
    assert next(s for s in found.symbols if s.name == "cons").used == (2, 4)
    assert "@pch" in source.completions(2, 1)
    line = len(TEXT.split("\n"))
    typed = ModelSource(TEXT + "x = @mo", "eviews")
    assert set(typed.completions(line, 8)) == {"@movav", "@movsum", "@movmax", "@movmin"}
    assert source.hover(2, 2).startswith("cons: endogenous variable")
    assert source.hover(3, 2).startswith("dlog(x[, n[, s]])")
    assert source.hover(4, 3).startswith("@identity y = ...")
    assert ModelSource("' cons = 1\n", "eviews").completions(1, 5) == []


def test_round_trips():
    data = _blank(_data(), "2003Q1")
    span = ("2003Q1", "2012Q4")
    built = ModelSource(TEXT, "eviews").build()
    for form in ("program", "text"):
        written = export_model(built, "eviews", form=form)
        again = read_eviews(written.text, params=VALUES) if form == "text" else ModelSource(written.text, "eviews").build()
        assert _residuals(again.model) == _residuals(built.model)
        a = built.model.solve(data, span, solver=Newton()).frame
        b = again.model.solve(data, span, solver=Newton()).frame
        assert np.allclose(a[["cons", "inv", "gdp", "inc"]], b[["cons", "inv", "gdp", "inc"]], rtol=0, atol=1e-12)
    # pymetrix -> EViews -> pymetrix: the template's steady state and responses kept (rbc's c renamed: reserved)
    for template in (templates.new_keynesian(), templates.gap_model(), templates.rbc()):
        out = export_model(template, "eviews")
        assert out.checked and out.difference < 1e-8, out.notes
        back = ModelSource(out.text, "eviews").export("pymetrix")
        assert back.checked
    rbc = export_model(templates.rbc(), "eviews")
    assert any("c as c_v" in n for n in rbc.notes) and any("measurement" in n for n in rbc.notes)
    nk = export_model(templates.new_keynesian(), "eviews")
    assert "m.append @innov" in nk.text or any("written as exogenous series" in n for n in nk.notes)
    # a backward-looking model written for Dynare: it has no shocks, and says so
    dynare = ModelSource(TEXT, "eviews").export("dynare")
    assert any("has no shocks" in n for n in dynare.notes) and "varexo" in dynare.text
    with pytest.raises(ModelError, match="no place in an IRIS"):
        ModelSource(TEXT, "eviews").export("iris")


def test_writer_refuses_what_eviews_has_not():
    model = Model([Equation("y ~ rho*y[-1] + step(x)", params=["rho"])], params={"rho": 0.5})
    with pytest.raises(ModelError, match="step"):
        export_model(model, "eviews")


# ---------------------------------------------------------------------------------------------
# FRB/US (the Federal Reserve's public package)


def _python_twin(text: str) -> str:
    """The Fed's Python version of an equation in pymetrix's syntax: x(-1) -> x[-1], ** -> ^ (functions log, exp, max)."""
    flat = " ".join(text.split()).replace("**", "^")
    return re.sub(r"\b(?!(?:log|exp|max|min)\b)([A-Za-z_]\w*)\s*\(\s*([-+]?\d+)\s*\)", lambda m: f"{m.group(1)}[{int(m.group(2)):+d}]", flat)


def _twins_agree(path: Path) -> tuple[int, float]:
    from pymetrix.expr.evaluate import evaluate
    from pymetrix.expr.parser import parse_equation
    from pymetrix.models.eviews import _xml_equations

    _, items = _xml_equations(path)
    vectors = {k.rpartition("_")[0] for it in items for tag in ("standard_equation", "mce_equation") if tag in it
               for k in it[tag]["coeffs"]}
    rng = np.random.default_rng(0)
    count, worst = 0, 0.0
    for it in items:
        for tag in ("standard_equation", "mce_equation"):
            if tag not in it:
                continue
            given = it[tag]
            lhs, rhs, used = xml_equation(given["eviews"], given["coeffs"], vectors)
            assert set(used) <= set(given["coeffs"])
            ours = parse_equation(f"{lhs} = {rhs}", list(given["coeffs"]))
            theirs = parse_equation(_python_twin(given["python"]), list(given["coeffs"]))
            draws: dict = {}

            def value(name, lag, draws=draws):
                return draws.setdefault((name, lag), rng.uniform(0.5, 1.5, size=40))

            a = evaluate(ours[0], value, given["coeffs"]) - evaluate(ours[2], value, given["coeffs"])
            b = evaluate(theirs[0], value, given["coeffs"]) - evaluate(theirs[2], value, given["coeffs"])
            assert np.array_equal(np.isfinite(a), np.isfinite(b)), it["name"]
            ok = np.isfinite(b)
            worst = max(worst, float(np.max(np.abs(a[ok] - b[ok]), initial=0.0)))
            count += 1
    return count, worst


def test_frbus_excerpt_equations_are_the_feds():
    count, worst = _twins_agree(FIXTURES / "frbus_excerpt.xml")
    assert count == 8 and worst < 1e-10  # six standard equations, two model-consistent-expectations versions
    read = read_eviews_xml(FIXTURES / "frbus_excerpt.xml")
    model = read.model
    assert model.endogenous == ("dmptmax", "delrff", "egfe", "wpsn", "zdivgr", "zyh")
    assert read.kinds["delrff"] == "Identity" and read.sectors["dmptmax"] == "Financial Sector"
    assert read.template.labels["delrff"] == "Federal funds rate, first diff"
    assert str(model.model_equations[0].residual).startswith("dmptmax - (max(dmptlur, dmptpi)")
    assert model.parameters.values()["y_zyh_1"] == pytest.approx(next(v for k, v in model.parameters.values().items() if k == "y_zyh_1"))
    mce = read_eviews_xml(FIXTURES / "frbus_excerpt.xml", mce=["mcap"])
    assert mce.model.has_leads and not model.has_leads and mce.mce["zdivgr"] == "mcap"


def _tracking(model: Model, frame: pd.DataFrame, start: pd.Period, periods: int) -> tuple[float, str]:
    """Add factors (_aerr) that make the baseline satisfy each equation, then the model solved with the endogenous
    variables blanked over the span: the largest relative gap to the baseline, and where."""
    from pymetrix.expr.evaluate import evaluate

    factors = [f"{e.dependent}_aerr" for e in model.model_equations]
    frame = pd.concat([frame, pd.DataFrame(0.0, index=frame.index, columns=factors)], axis=1)
    values = {f"{e.dependent}_aerr": evaluate(e.residual, lambda name, lag: frame[name].shift(-lag).to_numpy())
              for e in model.model_equations}
    frame = frame.assign(**values)
    blank = frame.copy()
    end = start + periods - 1
    blank.loc[start:end, list(model.endogenous)] = np.nan
    solved = model.solve(Dataset.from_frame(blank.rename_axis("time")), (str(start), str(end)), solver=Newton()).frame
    worst, where = 0.0, ""
    for v in model.endogenous:
        got, want = solved.loc[start:end, v].to_numpy(), frame.loc[start:end, v].to_numpy()
        gap = float(np.max(np.abs(got - want) / np.maximum(1.0, np.abs(want))))
        if not gap <= worst:
            worst, where = gap, v
    return worst, where


def test_frbus_excerpt_tracks_its_baseline():
    model = read_eviews_xml(FIXTURES / "frbus_excerpt.xml").model
    frame = pd.read_csv(FIXTURES / "frbus_longbase_excerpt.csv", comment="#")
    frame.index = pd.PeriodIndex(frame.pop("obs"), freq="Q")
    worst, where = _tracking(model, frame, pd.Period("2026Q2", "Q"), 6)
    assert worst < 1e-9, where


FRBUS = os.environ.get("PYMETRIX_FRBUS")  # the frbus_package folder (with mods/model.xml)
FRBUS_DATA = os.environ.get("PYMETRIX_FRBUS_DATA")  # the data_only_package folder (with LONGBASE.TXT, HISTDATA.TXT)


@pytest.mark.skipif(not FRBUS, reason="set PYMETRIX_FRBUS to the FRB/US package folder")
def test_frbus_every_equation_is_the_feds():
    path = Path(FRBUS) / "mods" / "model.xml"
    count, worst = _twins_agree(path)
    assert count == 309 and worst < 1e-10
    read = read_eviews_xml(path)
    assert len(read.model.endogenous) == 284 and len(read.equations) == 284
    assert sum(k == "Exogenous" for k in read.kinds.values()) == 367 - 284
    assert not read.model.has_leads and read_eviews_xml(path, mce=True).model.has_leads


@pytest.mark.skipif(not (FRBUS and FRBUS_DATA), reason="set PYMETRIX_FRBUS and PYMETRIX_FRBUS_DATA to the FRB/US packages")
def test_frbus_tracks_longbase():
    model = read_eviews_xml(Path(FRBUS) / "mods" / "model.xml").model
    frame = pd.read_csv(Path(FRBUS_DATA) / "LONGBASE.TXT")
    frame.columns = [c.lower() for c in frame.columns]
    frame.index = pd.PeriodIndex(frame.pop("obs"), freq="Q")
    history = pd.read_csv(Path(FRBUS_DATA) / "HISTDATA.TXT", usecols=[0]).iloc[-1, 0]
    worst, where = _tracking(model, frame, pd.Period(history, "Q") + 1, 40)
    assert worst < 1e-6, where


def test_editor_eviews_syntax():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from pymetrix.ui.model_editor_qt import LABELS, TEMPLATES, ModelEditorWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert ModelSource(TEMPLATES["eviews"], "eviews").check().ok and LABELS["eviews"] == "EViews (.prg / model text)"
    window = ModelEditorWindow()
    window.syntax.setCurrentIndex(window.syntax.findData("eviews"))
    assert window.editor.toPlainText() == TEMPLATES["eviews"] and window.editor.syntax == "eviews"
    window.editor.setPlainText(TEXT + "' a comment with @pch(x) and c(1)\nm.append y = \"text\"\n")
    window.editor.highlighter.rehighlight()
    window.refresh()
    window.editor.setPlainText(TEXT)
    window.refresh()
    assert window.last.ok
    other = window.export("eviews")
    assert other is not None and "m.append @identity gdp = cons + inv + gov" in other.editor.toPlainText()
    nk = ModelEditorWindow(text="parameters\n    rho = 0.8\nshocks\n    e = 1\nequations\n    x = rho*x[-1] + e\n")
    written = nk.export("eviews")
    assert written is not None and written.syntax.currentData() == "eviews" and "@innov x 1" in written.editor.toPlainText()
    for w in (window, other, nk, written):
        w.close()
    app.processEvents()


SSPACE = """' a trend and a cycle, as an EViews state-space object
sspace ss
ss.append @signal y = 0.5*x + trend + cycle + [var = c(3)^2]
ss.append @state trend = trend(-1) + [var = c(1)^2]
ss.append @state cycle = c(4)*cycle(-1) + c(5)*cycle1(-1) + ecyc
ss.append @state cycle1 = cycle(-1)
ss.append @ename ecyc
ss.append @evar var(ecyc) = c(2)^2
c(1) = 0.3
c(2) = 0.8
c(3) = 0.5
c(4) = 1.2
c(5) = -0.4
ss.ml
"""


def test_sspace_is_the_hand_written_state_space():
    from pymetrix.models.eviews import read_eviews_sspace
    from pymetrix.statespace import StateSpace

    read = read_eviews_sspace(SSPACE)
    hand = StateSpace(["trend", "cycle", "cycle1"],
                      ["trend = trend[-1] + e_trend", "cycle = c_4*cycle[-1] + c_5*cycle1[-1] + ecyc", "cycle1 = cycle[-1]"],
                      ["y - 0.5*x = trend + cycle + e_y"], {"e_trend": "c_1", "ecyc": "c_2", "e_y": "c_3"},
                      ["c_1", "c_2", "c_3", "c_4", "c_5"])
    rng = np.random.default_rng(0)
    x = rng.normal(size=80)
    data = Dataset.from_frame(pd.DataFrame({"y": np.cumsum(rng.normal(size=80)) + 0.5 * x, "x": x},
                                           index=pd.period_range("2000Q1", periods=80, freq="Q", name="time")))
    values = {"c_1": 0.3, "c_2": 0.8, "c_3": 0.5, "c_4": 1.2, "c_5": -0.4}
    assert {p.name: p.start for p in read.params} == values
    assert abs(read.loglike(data, values) - hand.loglike(data, values)) < 1e-10
    with pytest.raises(ModelError, match=r"line 3, column 23: a data series with an estimated coefficient"):
        read_eviews_sspace(SSPACE.replace("0.5*x", "c(6)*x"))
    with pytest.raises(ModelError, match=r"line 8, column 29: a variance is read as a coefficient squared"):
        read_eviews_sspace(SSPACE.replace("c(2)^2", "c(2)^2 + 1"))


FRBUS_SSPACE = os.environ.get("PYMETRIX_FRBUS_SSPACE")  # the state_space_package folder


@pytest.mark.skipif(not FRBUS_SSPACE, reason="set PYMETRIX_FRBUS_SSPACE to the FRB/US state-space package folder")
def test_frbus_state_space_program_read_as_far_as_it_maps():
    from pymetrix.models.eviews import _sspace
    from pymetrix.models.source import _place

    text = (Path(FRBUS_SSPACE) / "estimation_code.prg").read_text()
    _, as_is = _sspace(text)
    assert (_place(text, min(d.at for d in as_is if "substitution" in d.message)) == (18, 8))  # {%modname}: refused where it is
    named = text.replace("{%modname}", "ss")  # the object's name written out
    pieces, problems = _sspace(named)
    errors = sorted((_place(named, d.at), d.message.split(":")[0]) for d in problems if d.severity == "error")
    assert errors == [((1, 1), "'subroutine' is not read in a state-space object"), ((13, 1), "'call' is not read in a state-space object"),
                      ((66, 9), "a data series with an estimated coefficient (a regression in the signal) is not read"),
                      ((72, 13), "a data series with an estimated coefficient (a regression in the signal) is not read"),
                      ((78, 13), "a data series with an estimated coefficient (a regression in the signal) is not read"),
                      ((86, 4), "a data series with an estimated coefficient (a regression in the signal) is not read"),
                      ((113, 36), "a variance is read as a coefficient squared, beta(3)^2, or a number"),
                      ((170, 1), "'endsub' is not read in a state-space object")]
    assert len(pieces["transition"]) == 25 and len(pieces["measurement"]) == 4


def test_dated_functions_make_series_from_the_data():
    from pymetrix.models.eviews import eviews_series

    text = ("y = 0.1*@trend(60:1) + @trend(\"2000q2\") + @trend + @elem(x, \"2000q3\") + @seas(2) + @pchy(x) + "
            "(t <= 2001)\nwfcreate q 2000q1 2003q4\n")
    with pytest.raises(ModelError, match="no equations|wfcreate"):
        read_eviews("y = @pchy(x)\n")  # no frequency known
    model = read_eviews(text.replace("wfcreate q 2000q1 2003q4\n", ""), frequency="Q").model
    again = read_eviews("model m\nwfcreate q 2000q1 2003q4\nm.append " + text.split("\n")[0] + "\n").model
    assert _residuals(again) == _residuals(model)
    assert set(model.exogenous) == {"trend_1960_1", "trend_2000q2", "trend_start", "x_at_2000q3", "seas_2", "x", "t"}
    periods = pd.period_range("2000Q1", periods=16, freq="Q", name="time")
    x = np.arange(1.0, 17.0)
    data = eviews_series(model, Dataset.from_frame(pd.DataFrame({"x": x, "t": 2000 + np.arange(16) / 4}, index=periods)))
    frame = data.frame
    assert frame["trend_1960_1"].iloc[0] == 160 and frame["trend_2000q2"].iloc[0] == -1 and frame["trend_start"].iloc[3] == 3
    assert (frame["x_at_2000q3"] == 3.0).all() and frame["seas_2"].tolist() == [0.0, 1.0, 0.0, 0.0] * 4
    solved = model.solve(data, ("2001Q1", "2003Q4"), solver=Newton()).frame["y"].loc[pd.Period("2001Q1", "Q"):]
    f = frame.loc[pd.Period("2001Q1", "Q"):]
    expected = (0.1 * f["trend_1960_1"] + f["trend_2000q2"] + f["trend_start"] + 3.0 + f["seas_2"]
                + f["x"] / frame["x"].shift(4).loc[f.index] - 1 + (f["t"] <= 2001))
    assert np.allclose(solved.to_numpy(), expected.to_numpy(), atol=1e-12)


# ---------------------------------------------------------------------------------------------
# Brillet's book: Structural Econometric Modelling, Methodology and Tools with Applications under EViews. Opt-in: the
# book is its author's copyright, so its programs are read from the reader's own copies (PYMETRIX_BRILLET, a folder
# with brillet_basic.prg, brillet_small_model.txt and structmodel.txt, or the path of structmodel.txt itself).

BRILLET = os.environ.get("PYMETRIX_BRILLET")


def _brillet(name: str) -> Path:
    """The path of one of the book's files in PYMETRIX_BRILLET, or a skip when it is not there."""
    if not BRILLET:
        pytest.skip("set PYMETRIX_BRILLET to a folder with your copies of Brillet's book files")
    root = Path(BRILLET)
    path = root / name if root.is_dir() else (root if name == "structmodel.txt" else root.parent / name)
    if not path.is_file():
        pytest.skip(f"{name} is not in PYMETRIX_BRILLET ({BRILLET})")
    return path


def test_brillet_very_basic_model():
    """The book's first model (chapter 5, _fra_1): 17 equations of which several are "f*(...)" (a function of), f a
    scalar the book sets to 0 and to 1 to check the model's structure; solved by Newton with f given."""
    text = _brillet("brillet_basic.prg").read_text()
    found = ModelSource(text, "eviews").check()
    assert [(d.line, d.column) for d in found.errors] == [(6, 8)] and "scalar f is declared without a value" in found.errors[0].message
    built = read_eviews(text, params={"f": 0.2})
    model = built.model
    assert built.name == "_fra_1" and len(model.endogenous) == 17
    assert set(model.exogenous) == {"depr", "gd", "lg", "r_ic", "r_icq", "r_ih", "r_rhiq", "sr", "wd", "wr"}
    assert model.model_equations[-1].dependent == "q" and str(model.model_equations[-1].residual) == "q - (fd + x - m)"
    rng = np.random.default_rng(3)
    periods = pd.period_range("2000Q1", periods=12, freq="Q", name="time")
    frame = pd.DataFrame({v: rng.uniform(0.05, 0.3, 12) for v in model.exogenous}, index=periods)
    for v in model.endogenous:
        frame[v] = np.nan
    frame.iloc[0, frame.columns.get_indexer(["k"])] = 10.0
    solved = model.solve(Dataset.from_frame(frame), ("2000Q2", "2002Q4"), solver=Newton()).frame.iloc[1:]
    assert np.allclose(solved["q"] + solved["m"], solved["fd"] + solved["x"], atol=1e-10)
    assert np.allclose(solved["k"], solved["k"].shift(1).fillna(10.0) * (1 - frame["depr"].iloc[1:]) + solved["i"], atol=1e-10)
    assert np.allclose(solved["ur"], solved["q"] / solved["cap"], atol=1e-10)


def test_brillet_small_estimated_model():
    """The book's estimated small model of France (section 6.3.7): 20 equations with their coefficients, @pch, @trend,
    @elem and comparisons with the time variable t; read without refusals, its structure, and a Newton solve on
    synthetic data whose solution satisfies every equation."""
    from pymetrix.expr.evaluate import evaluate
    from pymetrix.models.eviews import eviews_series

    text = _brillet("brillet_small_model.txt").read_text()
    undeclared = read_eviews(text).model  # p_x(1), a vector EViews knows from the workfile: without it, a lead of a series
    assert "p_x" in undeclared.exogenous and undeclared.has_leads
    built = read_eviews(text, params={"p_x(1)": 1.0})  # assumed: its value is the VAR's, not in the text
    model = built.model
    assert len(model.endogenous) == 20 and not model.has_leads
    assert set(model.exogenous) == {"compm", "ct", "dr", "ec_ci", "ec_i", "ec_le", "ec_m", "ec_x", "gd", "lg", "pk", "r_ih", "r_rhiq",
                                    "sr", "t", "t1", "t2", "trend_1960_1", "ur_star", "wd", "wr"}
    residual = {e.dependent: str(e.residual) for e in model.model_equations}
    assert "179" in residual["res_x"] and "t <= 2004" in residual["res_x"]
    periods = pd.period_range("1998Q1", periods=40, freq="Q", name="time")
    n = len(periods)
    growth = 1.005 ** np.arange(n)
    exo = {"compm": 1.0, "ct": 0.01, "dr": 0.02, "ec_ci": 0.0, "ec_i": 0.0, "ec_le": 0.0, "ec_m": 0.0, "ec_x": 0.0, "gd": 143.7 * growth,
           "lg": 5.0, "pk": 0.25, "r_ih": 0.05, "r_rhiq": 0.1, "sr": 0.15, "t": 1998 + np.arange(n) / 4, "t1": 2001.0,
           "t2": 2003.0, "ur_star": 0.85, "wd": 100 * growth, "wr": 4.0}
    frame = pd.DataFrame(exo, index=periods)
    start = {"cap": 300.0, "ur": 0.85, "q": 255.0, "m": 60.0, "fd": 215.0, "x": 100.0, "i": 30.0, "k": 1200.0, "ic": 2.5,
             "prle_t": 18000.0, "led": 0.014, "le": 0.014, "lt": 5.0, "rhi": 45.0, "co": 38.0, "ih": 2.3, "ci": 1.0, "td": 218.0,
             "res_m": 0.0, "res_x": 0.0}
    for v, value in start.items():
        frame[v] = value * growth
    frame = eviews_series(model, Dataset.from_frame(frame)).frame
    for e in model.model_equations:  # the residuals (ec_) that make the synthetic history satisfy the estimated equations
        error = {"i": "ec_i", "le": "ec_le", "ci": "ec_ci", "m": "ec_m", "x": "ec_x"}.get(e.dependent)
        if error:
            frame[error] = pd.Series(evaluate(e.residual, lambda name, lag: frame[name].shift(-lag).to_numpy()), frame.index).bfill()
    data = Dataset.from_frame(frame)
    span = ("2002Q1", "2004Q4")
    solved = model.solve(data, span, solver=Newton()).frame
    window = solved.loc[pd.Period(span[0], "Q"):pd.Period(span[1], "Q")]
    assert np.isfinite(window[list(model.endogenous)].to_numpy()).all()
    full = data.frame.copy()
    full.loc[window.index, list(model.endogenous)] = window[list(model.endogenous)]
    for e in model.model_equations:  # every equation holds at the solution
        values = evaluate(e.residual, lambda name, lag: full[name].shift(-lag).to_numpy())
        assert np.max(np.abs(values[full.index.get_indexer(window.index)])) < 1e-8, e.name


def test_brillet_larger_model_from_the_book():
    """The book's larger model (chapter 11, _mod_1: every "_mod_1.append" line of its listing, with f and @pchy): read
    without refusals once its frequency and f are given, 83 equations, compiled."""
    lines = _brillet("structmodel.txt").read_text().split("\n")
    start = next(i for i, line in enumerate(lines) if line.strip() == "model _mod_1")
    equations = [line.strip() for line in lines[start:start + 600] if re.match(r"\s*_mod_1\.append ", line)]
    text = "model _mod_1\nscalar f\n" + "\n".join(equations) + "\n"
    found = ModelSource(text, "eviews").check()
    assert {d.message.split(":")[0] for d in found.errors} == {"the scalar f is declared without a value", "@pchy needs the data's frequency"}
    model = read_eviews(text, params={"f": 1.0}, frequency="Q").model
    assert len(equations) == len(model.endogenous) == 83 and len(model.exogenous) == 40
