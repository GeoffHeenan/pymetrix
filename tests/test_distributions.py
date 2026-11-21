"""Distribution functions in the expression language, checked independently.

- Every pdf (pmf), cdf and inverse cdf against scipy.stats at many points and parameter values, MATLAB's
  parameterisations mapped by hand (gamma shape and scale, exponential mean, lognormal mu and sigma of log x, uniform
  a and b): 1e-12 relative (and 1e-300 absolute, for values that underflow); erf and erfc against scipy.special.
- inv(cdf(x)) = x for every continuous family, to 1e-9 (1e-8 for t with SciPy before 1.13, whose t.ppf is less exact).
- Every derivative the language offers (in x, and in the parameters where it offers one) against central finite
  differences, interpreted and compiled; derivatives it does not offer are refused with a message.
- The interpreter and the compiled solver functions agree; Series stay Series; missing values stay missing.
- A model with a probit-style rule p = normcdf(a + b*y) solved and linearised: its decision rules against the
  closed form (the fixed point of g = rho/(1 - 0.2 c g), c = b*phi(a + b*y*)); the same model read from a Dynare
  file (with normcdf in a parameter value too) gives the same rules; the model editor's checker accepts the functions.
- Notation (Phi, phi, Phi^-1) and the Dynare/IRIS writers (normcdf, normpdf and erf written; others refused).
"""

import numpy as np
import pandas as pd
import pytest
import scipy
from packaging.version import Version
from scipy import special, stats

from pymetrix import Dataset, Equation, Model, read_mod
from pymetrix.expr import Coef, ExpressionError, Name, diff, evaluate, parse_expression
from pymetrix.expr.compile import compile_exprs
from pymetrix.expr.functions import DISTRIBUTIONS

X = np.linspace(-3.0, 3.0, 25)
POS = np.linspace(0.05, 12.0, 30)
UNIT = np.linspace(0.01, 0.99, 30)
P = np.linspace(0.001, 0.999, 41)
K = np.arange(0, 16).astype(float)

# (name, points, parameter sets, reference(x, *params))
CASES = [
    ("normpdf", X, [(), (0.5,), (0.5, 2.0), (-1.0, 0.3)], lambda x, mu=0.0, s=1.0: stats.norm.pdf(x, mu, s)),
    ("normcdf", X, [(), (0.5,), (0.5, 2.0), (-1.0, 0.3)], lambda x, mu=0.0, s=1.0: stats.norm.cdf(x, mu, s)),
    ("norminv", P, [(), (0.5, 2.0), (-1.0, 0.3)], lambda p, mu=0.0, s=1.0: stats.norm.ppf(p, mu, s)),
    ("tpdf", X, [(1.0,), (3.5,), (30.0,)], lambda x, v: stats.t.pdf(x, v)),
    ("tcdf", X, [(1.0,), (3.5,), (30.0,)], lambda x, v: stats.t.cdf(x, v)),
    ("tinv", P, [(1.0,), (3.5,), (30.0,)], lambda p, v: stats.t.ppf(p, v)),
    ("chi2pdf", POS, [(1.0,), (4.0,), (9.5,)], lambda x, v: stats.chi2.pdf(x, v)),
    ("chi2cdf", POS, [(1.0,), (4.0,), (9.5,)], lambda x, v: stats.chi2.cdf(x, v)),
    ("chi2inv", P, [(1.0,), (4.0,), (9.5,)], lambda p, v: stats.chi2.ppf(p, v)),
    ("fpdf", POS, [(1.0, 5.0), (4.0, 20.0), (10.0, 3.0)], lambda x, a, b: stats.f.pdf(x, a, b)),
    ("fcdf", POS, [(1.0, 5.0), (4.0, 20.0), (10.0, 3.0)], lambda x, a, b: stats.f.cdf(x, a, b)),
    ("finv", P, [(1.0, 5.0), (4.0, 20.0), (10.0, 3.0)], lambda p, a, b: stats.f.ppf(p, a, b)),
    ("gampdf", POS, [(0.5, 1.0), (2.0, 3.0), (7.5, 0.4)], lambda x, a, b: stats.gamma.pdf(x, a, scale=b)),
    ("gamcdf", POS, [(0.5, 1.0), (2.0, 3.0), (7.5, 0.4)], lambda x, a, b: stats.gamma.cdf(x, a, scale=b)),
    ("gaminv", P, [(0.5, 1.0), (2.0, 3.0), (7.5, 0.4)], lambda p, a, b: stats.gamma.ppf(p, a, scale=b)),
    ("betapdf", UNIT, [(0.5, 0.5), (2.0, 5.0), (8.0, 1.5)], lambda x, a, b: stats.beta.pdf(x, a, b)),
    ("betacdf", UNIT, [(0.5, 0.5), (2.0, 5.0), (8.0, 1.5)], lambda x, a, b: stats.beta.cdf(x, a, b)),
    ("betainv", P, [(0.5, 0.5), (2.0, 5.0), (8.0, 1.5)], lambda p, a, b: stats.beta.ppf(p, a, b)),
    ("lognpdf", POS, [(), (0.5, 0.8), (1.0, 1.5)], lambda x, mu=0.0, s=1.0: stats.lognorm.pdf(x, s, scale=np.exp(mu))),
    ("logncdf", POS, [(), (0.5, 0.8), (1.0, 1.5)], lambda x, mu=0.0, s=1.0: stats.lognorm.cdf(x, s, scale=np.exp(mu))),
    ("logninv", P, [(), (0.5, 0.8), (1.0, 1.5)], lambda p, mu=0.0, s=1.0: stats.lognorm.ppf(p, s, scale=np.exp(mu))),
    ("unifpdf", X, [(), (-1.0, 2.5)], lambda x, a=0.0, b=1.0: stats.uniform.pdf(x, a, b - a)),
    ("unifcdf", X, [(), (-1.0, 2.5)], lambda x, a=0.0, b=1.0: stats.uniform.cdf(x, a, b - a)),
    ("unifinv", P, [(), (-1.0, 2.5)], lambda p, a=0.0, b=1.0: stats.uniform.ppf(p, a, b - a)),
    ("exppdf", POS, [(), (2.5,), (0.3,)], lambda x, mu=1.0: stats.expon.pdf(x, scale=mu)),
    ("expcdf", POS, [(), (2.5,), (0.3,)], lambda x, mu=1.0: stats.expon.cdf(x, scale=mu)),
    ("expinv", P, [(), (2.5,), (0.3,)], lambda p, mu=1.0: stats.expon.ppf(p, scale=mu)),
    ("poisspdf", K, [(0.5,), (4.0,), (11.2,)], lambda k, lam: stats.poisson.pmf(k, lam)),
    ("poisscdf", K, [(0.5,), (4.0,), (11.2,)], lambda k, lam: stats.poisson.cdf(k, lam)),
    ("binopdf", K, [(15.0, 0.2), (20.0, 0.65), (40.0, 0.05)], lambda k, n, p: stats.binom.pmf(k, n, p)),
    ("binocdf", K, [(15.0, 0.2), (20.0, 0.65), (40.0, 0.05)], lambda k, n, p: stats.binom.cdf(k, n, p)),
    ("poissinv", P, [(0.5,), (4.0,), (11.2,)], lambda p, lam: stats.poisson.ppf(p, lam)),
    ("binoinv", P, [(15.0, 0.2), (20.0, 0.65), (40.0, 0.05)], lambda p, n, pr: stats.binom.ppf(p, n, pr)),
]


def _value(text, x):
    """``text`` (a function of x) evaluated over an array of x by the interpreter."""
    return np.asarray(evaluate(parse_expression(text), lambda name, lag: x), dtype=float)


def _call(fn, params):
    return f"{fn}({', '.join(['x', *map(repr, params)])})"


def test_every_function_is_checked():
    assert {c[0] for c in CASES} == set(DISTRIBUTIONS)


@pytest.mark.parametrize("fn, points, sets, reference", CASES, ids=[c[0] for c in CASES])
def test_values_against_scipy(fn, points, sets, reference):
    for params in sets:
        got = _value(_call(fn, params), points)
        want = reference(points, *params)
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-300, err_msg=f"{fn}{params}")


def test_erf_and_erfc():
    np.testing.assert_allclose(_value("erf(x)", X), special.erf(X), rtol=1e-12)
    np.testing.assert_allclose(_value("erfc(x)", X), special.erfc(X), rtol=1e-12)
    np.testing.assert_allclose(_value("normcdf(x)", X), 0.5 * special.erfc(-X / np.sqrt(2)), rtol=1e-12)


@pytest.mark.parametrize("family, points, params", [
    ("norm", X, (0.5, 2.0)), ("t", X, (3.5,)), ("chi2", POS, (4.0,)), ("f", POS, (4.0, 20.0)), ("gam", POS, (2.0, 3.0)),
    ("beta", UNIT, (2.0, 5.0)), ("logn", POS, (0.5, 0.8)), ("unif", np.linspace(-0.9, 2.4, 20), (-1.0, 2.5)), ("exp", POS, (2.5,)),
])
def test_inverse_of_the_cdf(family, points, params):
    args = "".join(f", {p!r}" for p in params)
    back = _value(f"{family}inv({family}cdf(x{args}){args})", points)
    # tinv is SciPy's t.ppf, accurate to about 2e-9 before SciPy 1.13 (2e-11 since): the oldest SciPy supported is 1.11
    old_t = family == "t" and Version(scipy.__version__) < Version("1.13")
    np.testing.assert_allclose(back, points, rtol=1e-8 if old_t else 1e-9, atol=1e-12)


# derivatives: (expression in x, a and b; the point; parameter values); a and b are parameters
DERIVATIVES = [
    ("normpdf(x, a, b)", 0.4, {"a": 0.3, "b": 1.7}), ("normcdf(x, a, b)", 0.4, {"a": 0.3, "b": 1.7}),
    ("norminv(x, a, b)", 0.3, {"a": 0.3, "b": 1.7}), ("normcdf(a + b*x)", -0.2, {"a": 0.1, "b": 2.0}),
    ("normpdf(x^2)", 0.7, {}), ("norminv(x)", 0.8, {}),
    ("tpdf(x, 4.5)", 0.7, {}), ("tcdf(x, 4.5)", 0.7, {}), ("tinv(x, 4.5)", 0.7, {}),
    ("chi2pdf(x, 3.5)", 2.1, {}), ("chi2cdf(x, 3.5)", 2.1, {}), ("chi2inv(x, 3.5)", 0.6, {}),
    ("fpdf(x, 4, 9)", 1.3, {}), ("fcdf(x, 4, 9)", 1.3, {}), ("finv(x, 4, 9)", 0.4, {}),
    ("gampdf(x, 2.5, b)", 1.9, {"b": 1.4}), ("gamcdf(x, 2.5, b)", 1.9, {"b": 1.4}), ("gaminv(x, 2.5, b)", 0.35, {"b": 1.4}),
    ("betapdf(x, 2.5, 4)", 0.3, {}), ("betacdf(x, 2.5, 4)", 0.3, {}), ("betainv(x, 2.5, 4)", 0.3, {}),
    ("lognpdf(x, a, b)", 1.6, {"a": 0.2, "b": 0.7}), ("logncdf(x, a, b)", 1.6, {"a": 0.2, "b": 0.7}),
    ("logninv(x, a, b)", 0.6, {"a": 0.2, "b": 0.7}),
    ("unifpdf(x, a, b)", 0.4, {"a": -1.0, "b": 2.0}), ("unifcdf(x, a, b)", 0.4, {"a": -1.0, "b": 2.0}),
    ("unifinv(x, a, b)", 0.4, {"a": -1.0, "b": 2.0}),
    ("exppdf(x, a)", 1.2, {"a": 2.2}), ("expcdf(x, a)", 1.2, {"a": 2.2}), ("expinv(x, a)", 0.7, {"a": 2.2}),
    ("poisspdf(3, x)", 2.7, {}), ("poisscdf(3, x)", 2.7, {}), ("binopdf(4, 12, x)", 0.3, {}), ("binocdf(4, 12, x)", 0.3, {}),
    ("erf(x)", 0.6, {}), ("erfc(2*x)", 0.3, {}),
]


@pytest.mark.parametrize("text, at, values", DERIVATIVES, ids=[d[0] for d in DERIVATIVES])
def test_derivatives_against_finite_differences(text, at, values):
    expr = parse_expression(text, params=list(values))
    for wrt, h in [(Name("x"), 1e-6)] + [(Coef(c), 1e-6) for c in values]:
        def f(v, wrt=wrt):
            point = {"x": at, **values}
            point[wrt.id] = v
            return float(evaluate(expr, lambda name, lag: point[name], {k: point[k] for k in values}))

        start = at if isinstance(wrt, Name) else values[wrt.id]
        numeric = (f(start + h) - f(start - h)) / (2 * h)
        d = diff(expr, wrt)
        point = {"x": at, **values}
        exact = float(evaluate(d, lambda name, lag: point[name], values))
        assert exact == pytest.approx(numeric, rel=1e-6, abs=1e-8), (text, str(wrt), str(d))
        compiled = compile_exprs([d], tuple(values))
        got = compiled([point[n] for n, _ in compiled.refs], [values[c] for c in values])[0]
        assert float(got) == pytest.approx(exact, rel=1e-12, abs=1e-15)


def test_derivatives_not_offered_are_refused():
    for text, wrt in [("tcdf(x, v)", "v"), ("chi2pdf(x, v)", "v"), ("finv(x, v, 3)", "v"), ("gamcdf(x, v, 2)", "v"),
                      ("betapdf(x, 2, v)", "v"), ("binopdf(2, v, 0.3)", "v")]:
        e = parse_expression(text, params=["v"])
        with pytest.raises(ExpressionError, match="cannot be differentiated with respect to"):
            diff(e, Coef(wrt))
        assert float(evaluate(diff(e, Coef("q")), lambda n, lag: 1.0, {"v": 3.0})) == 0.0  # other arguments are fine


def test_parsing_printing_and_arguments():
    e = parse_expression("normcdf(a*x[-1] + b, mu, 2) - gaminv(0.5, 2, 3)", params=["a", "b", "mu"])
    assert parse_expression(str(e), params=["a", "b", "mu"]) == e
    assert e.refs() == {Name("x", -1)} and e.params() == {"a", "b", "mu"}
    assert str(e.shift(-1)) == "normcdf(a*x[-2] + b, mu, 2) - gaminv(0.5, 2, 3)"
    with pytest.raises(ExpressionError, match=r"tcdf\(\) takes 2 arguments: tcdf\(x, df\)"):
        parse_expression("tcdf(x)")
    with pytest.raises(ExpressionError, match=r"normcdf\(\) takes 1 to 3 arguments"):
        parse_expression("normcdf(x, 0, 1, 2)")
    assert float(evaluate(parse_expression("norminv(0.975)"), lambda n, lag: 0.0)) == pytest.approx(1.959963984540054, rel=1e-14)


def test_series_missing_values_and_invalid_parameters():
    data = Dataset.from_frame(pd.DataFrame({"z": [-1.0, np.nan, 0.5, 2.0]}, index=pd.period_range("2000Q1", periods=4, freq="Q")))
    out = data.with_series(p="normcdf(z)", q="poisspdf(z + 1, 2)", s="normcdf(z, 0, -1)").frame
    np.testing.assert_allclose(out["p"].to_numpy(), stats.norm.cdf([-1.0, np.nan, 0.5, 2.0]), rtol=1e-12)
    assert np.isnan(out["q"].iloc[1]) and out["q"].iloc[0] == pytest.approx(stats.poisson.pmf(0, 2), rel=1e-12)
    assert out["s"].isna().all()  # a negative standard deviation: missing, as in MATLAB
    assert isinstance(data.eval("tcdf(z, 3)"), pd.Series)


def test_a_model_with_a_probit_rule_is_solved_and_linearised():
    rho, a, b = 0.5, -0.5, 2.0
    model = Model([Equation("y ~ rho*y[-1] + 0.2*p[+1] + e", params=["rho"]), Equation("p ~ normcdf(a + b*y)", params=["a", "b"])],
                  params=dict(rho=rho, a=a, b=b))
    ss = model.steady_state()
    y = ss["y"]
    assert y == pytest.approx(0.2 * stats.norm.cdf(a + b * y) / (1 - rho), rel=1e-10)
    rules = model.linearize(shocks=["e"]).solve().rules().to_frame().set_index("")
    c = b * stats.norm.pdf(a + b * y)
    g = (1 - np.sqrt(1 - 4 * 0.2 * c * rho)) / (2 * 0.2 * c)  # the stable root of 0.2 c g^2 - g + rho = 0
    assert rules.loc["y", "y[-1]"] == pytest.approx(g, rel=1e-8)
    assert rules.loc["p", "y[-1]"] == pytest.approx(c * g, rel=1e-8)
    assert rules.loc["y", "e"] == pytest.approx(g / rho, rel=1e-8)


def test_a_model_without_an_offered_derivative_is_refused():
    model = Model([Equation("y ~ r*y[-1] + 0.1*tcdf(0.5, 5 + y[-1]) + e", params=["r"])], params={"r": 0.5})
    with pytest.raises(ExpressionError, match="cannot be differentiated"):
        model.linearize(shocks=["e"]).solve()


def test_a_dynare_file_with_normcdf_is_read_and_solved():
    dyn = read_mod("""
        var y p; varexo e; parameters rho a b;
        rho = 0.5; a = -0.5; b = normcdf(0)*4;   // b = 2
        model;
        y = rho*y(-1) + 0.2*p(+1) + e;
        p = normcdf(a + b*y, 0, 1);
        end;
        shocks; var e; stderr 0.1; end;
    """)
    assert dyn.params["b"] == pytest.approx(2.0, rel=1e-15)
    by_hand = Model([Equation("y ~ rho*y[-1] + 0.2*p[+1] + e", params=["rho"]), Equation("p ~ normcdf(a + b*y)", params=["a", "b"])],
                    params=dict(rho=0.5, a=-0.5, b=2.0)).linearize(shocks=["e"]).solve()
    np.testing.assert_allclose(dyn.solve().irf(8).values, by_hand.irf(8).values, rtol=1e-10, atol=1e-14)


def test_the_model_editor_accepts_the_functions():
    from pymetrix.models.source import ModelSource

    text = """
    parameters
        rho = 0.5
        nu = 4
    shocks
        e
    equations
        y = rho*y[-1] + 0.1*tcdf(y[+1], nu) + 0.1*movavg(y, 3) + e
        p = normcdf(y) + erfc(y)
    """
    check = ModelSource(text).check()
    assert not [d for d in check.diagnostics if d.severity == "error"], check.diagnostics


def test_notation_and_writers():
    from pymetrix.notation import algebraic

    eq = Equation("p ~ normcdf(a + b*y) + normpdf(y) + norminv(q) + tcdf(y, 4)", params=["a", "b"])
    latex = algebraic(eq, style="latex")
    assert "\\Phi\\left(" in latex and "\\phi\\left(" in latex and "\\Phi^{-1}\\left(" in latex
    assert "\\operatorname{tcdf}\\left(" in latex
    assert "Φ(" in algebraic(eq) and "Φ⁻¹(" in algebraic(eq)

    from pymetrix import ModelSource
    from pymetrix.models.solvers import ModelError

    text = "parameters\n    rho = 0.5\nshocks\n    e\nequations\n    y = rho*y[-1] + 0.1*normcdf(y[-1], 0, 2) + 0.1*erf(y[+1]) + {} + e\n"
    dynare = ModelSource(text.format("normpdf(y)")).export("dynare").text
    assert "normcdf(y(-1), 0, 2)" in dynare and "erf(y(+1))" in dynare and "normpdf(y)" in dynare
    assert "normcdf(y{-1}, 0, 2)" in ModelSource(text.format("0")).export("iris").text
    for bad in ["tcdf(y[-1], 4)", "movavg(y, 3)", "erfc(y)"]:
        with pytest.raises(ModelError, match="no equivalent in Dynare or IRIS"):
            ModelSource(text.format(bad)).export("dynare")
