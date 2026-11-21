import numpy as np
import pytest

from pymetrix.expr import Coef, ExpressionError, Name, diff, evaluate, parse_equation, parse_expression


def test_lags_are_pushed_onto_names():
    e = parse_expression("(log(c) - log(yd))[-1]")
    assert e.refs() == {Name("c", -1), Name("yd", -1)}


def test_difference_functions_expand():
    assert parse_expression("d(x)") == parse_expression("x - x[-1]")  # expanded into primitives...
    assert parse_expression("dlog(x)") == parse_expression("log(x) - log(x[-1])")
    assert str(parse_expression("dlog(x)")) == "dlog(x)" and str(parse_expression("dlog(x)").shift(-1)) == "dlog(x)[-1]"  # ...but printed as written
    assert parse_expression("d(x, 2)").refs() == {Name("x", 0), Name("x", -1), Name("x", -2)}


def test_declared_names_are_parameters():
    lhs, op, rhs = parse_equation("y ~ a + b*x", params=["a", "b"])
    assert op == "~"
    assert rhs.params() == {"a", "b"}
    assert rhs.variables() == {"x"}


def test_round_trip_through_str():
    for text in ["a*x^2 - b/(1 + z)", "-x^2", "(-x)^2", "y[-1] + exp(2*x[+1])"]:
        e = parse_expression(text, ["a", "b"])
        assert parse_expression(str(e), ["a", "b"]) == e


@pytest.mark.parametrize("text", ["b[-1]", "foo(x)", "x[-1.5]", "x +", "y = x"])
def test_errors(text):
    with pytest.raises(ExpressionError):
        parse_expression(text, ["b"])


def test_derivatives_match_finite_differences():
    e = parse_expression("b*x^2*log(z) + exp(b*z)/x - sqrt(x*z)", ["b"])
    point = {"x": 1.3, "z": 2.1}
    resolve = lambda name, lag: point[name]  # noqa: E731
    for wrt in (Name("x"), Name("z"), Coef("b")):
        h = 1e-6

        def at(delta):
            if isinstance(wrt, Coef):
                return evaluate(e, resolve, {"b": 0.7 + delta})
            moved = {**point, wrt.id: point[wrt.id] + delta}
            return evaluate(e, lambda n, lag: moved[n], {"b": 0.7})

        numeric = (at(h) - at(-h)) / (2 * h)
        assert np.isclose(evaluate(diff(e, wrt), resolve, {"b": 0.7}), numeric, rtol=1e-6)


# -- conditions: comparisons, and, or, not --


def test_conditions_parse_with_the_usual_precedence():
    from pymetrix.expr.nodes import BinOp, Call

    e = parse_expression("not u > 5 and infl - 2 >= 0 or x[-1] == 1")
    assert isinstance(e, BinOp) and e.op == "or"  # or binds loosest, then and, then not, then comparisons
    assert e.left.op == "and" and isinstance(e.left.left, Call) and e.left.left.fn == "not"
    assert e.left.left.args[0].op == ">" and e.left.right.op == ">=" and e.left.right.left.op == "-"  # arithmetic first
    assert e.right.left == Name("x", -1)
    for text in ("gy < 0", "u >= 5 and infl != 2", "not(a <= b) or c == 1", "a + b > 2*c", "(a or b) and c"):
        assert parse_expression(str(parse_expression(text))) == parse_expression(text)  # printing round-trips
    assert str(parse_expression("(a or b) and c")) == "(a or b) and c"
    with pytest.raises(ExpressionError, match="join comparisons with 'and'"):
        parse_expression("0 < x < 1")
    with pytest.raises(ExpressionError, match="joins two conditions"):
        parse_expression("and x")
    lhs, sign, rhs = parse_equation("d = (gy < 0)")  # = is still an identity; == a comparison
    assert sign == "=" and rhs.op == "<"


def test_conditions_are_one_zero_and_missing_stays_missing():
    a = np.array([1.0, 5.0, np.nan, -2.0])
    b = np.array([2.0, 5.0, 1.0, np.nan])
    values = {"a": a, "b": b}

    def ev(text):
        return evaluate(parse_expression(text), lambda name, lag: values[name])

    by_hand = {"a < b": [1, 0, np.nan, np.nan], "a <= b": [1, 1, np.nan, np.nan], "a > b": [0, 0, np.nan, np.nan],
               "a >= b": [0, 1, np.nan, np.nan], "a == b": [0, 1, np.nan, np.nan], "a != b": [1, 0, np.nan, np.nan],
               "a > 0 and b > 0": [1, 1, np.nan, np.nan], "a > 3 or b > 3": [0, 1, np.nan, np.nan],
               "not a > 3": [1, 0, np.nan, 1]}
    for text, expected in by_hand.items():
        np.testing.assert_array_equal(ev(text), np.array(expected, float), err_msg=text)
    assert ev("2 > 1") == 1.0 and ev("not 0") == 1.0


def test_conditions_compile_and_differentiate():
    from pymetrix.expr.compile import compile_exprs

    texts = ["a < b", "a >= b", "a == b", "a > 0 and b > 0", "a > 3 or b > 3", "not a > 3", "x * (a > 0)"]
    exprs = [parse_expression(t) for t in texts]
    compiled = compile_exprs(exprs)
    data = {"a": np.array([1.0, 5.0, np.nan, -2.0]), "b": np.array([2.0, 5.0, 1.0, np.nan]), "x": np.array([3.0, 4.0, 5.0, 6.0])}
    outs = compiled([data[name] for name, _ in compiled.refs])
    for e, out in zip(exprs, outs):  # the compiled function agrees with the interpreter, missing values included
        np.testing.assert_array_equal(out, evaluate(e, lambda name, lag: data[name]), err_msg=str(e))
    assert diff(parse_expression("a < b"), Name("a")) == parse_expression("0")  # flat, as step() is
    assert str(diff(parse_expression("x * (a > 0)"), Name("x"))) == "a > 0"  # the dummy multiplies through


def test_conditions_make_dummies_and_regressors():
    from pymetrix import Dataset, Equation

    data = Dataset.example("sample_macro").with_series(gy="400*dlog(y)")
    made = data.with_series(recession="gy < 0", boom="gy > 5 and gy[-1] > 5")
    gy = data.frame["gy"]
    expected = np.where(gy.isna(), np.nan, (gy < 0).astype(float))
    np.testing.assert_array_equal(made.frame["recession"].to_numpy(), expected)
    boom = np.where(gy.isna() | gy.shift(1).isna(), np.nan, ((gy > 5) & (gy.shift(1) > 5)).astype(float))
    np.testing.assert_array_equal(made.frame["boom"].to_numpy(), boom)
    inline = Equation("c ~ y + (gy < 0)").fit(data)  # a dummy written in the equation
    stored = Equation("c ~ y + recession").fit(made)
    assert np.allclose(inline.params.values, stored.params.values)
    figure = made.plot("gy", shade="recession")  # the shading that needed a dummy made outside formulas
    assert figure.shade and all(made.frame["recession"].loc[s:e].eq(1).all() for s, e, _ in figure.shade)


def test_conditions_in_notation():
    from pymetrix.notation import expression

    e = parse_expression("u >= 5 and not infl != 2")
    assert expression(e) == "uₜ ≥ 5 and not (inflₜ ≠ 2)"
    assert expression(e, "latex") == r"u_t \ge 5 \;\text{and}\; \neg \left(\mathit{infl}_t \ne 2\right)"


def test_compiled_shared_subexpressions_and_plain_numbers():
    """Repeated subtrees are worked out once (the source shrinks) with the interpreter's results; plain numbers take
    Python's arithmetic, with NumPy's results at the edges (division by zero, logs and roots of negatives, overflow,
    powers of negatives, NaN in a maximum) and to rounding elsewhere, checked value by value against arrays."""
    from pymetrix.expr.compile import _SCALAR, compile_exprs

    texts = ["(a + b)^2 / (a + b) + log(a + b) * exp(a - b)", "(a + b)^2 * c", "max(a, b) + min(a, c) + sqrt(b) + abs(c)",
             "a / b", "log(a) + log(b)", "b^0.5", "exp(1000 * a)", "(a + b)^c"]
    exprs = [parse_expression(t) for t in texts]
    compiled = compile_exprs(exprs)
    assert compiled.source.count("(v[0] + v[1])") == 1  # once, into a temporary
    cases = [(1.5, 2.0, -0.5), (0.0, 0.0, 1.0), (-1.0, -2.0, 0.5), (np.nan, 1.0, 2.0), (2.0, np.inf, -1.0), (-0.0, 3.0, 0.0)]
    columns = [np.array(c) for c in zip(*cases)]
    order = [{"a": 0, "b": 1, "c": 2}[name] for name, _ in compiled.refs]
    with np.errstate(all="ignore"):
        arrays = compiled([columns[k] for k in order])
        for i, case in enumerate(cases):
            scalars = compiled([case[k] for k in order])
            assert all(type(s) is not np.ndarray or s.ndim == 0 for s in scalars)
            for e, s, arr in zip(exprs, scalars, arrays):
                np.testing.assert_allclose(s, arr[i], rtol=1e-15, err_msg=f"{e} at {case}")
                np.testing.assert_allclose(s, evaluate(e, lambda name, lag, c=case: c["abc".index(name)]), rtol=1e-15)
    assert np.isnan(_SCALAR.power(-8.0, 1 / 3)) and _SCALAR.divide(1.0, 0.0) == np.inf and np.isnan(_SCALAR.maximum(np.nan, 1.0))


def test_walk_is_depth_first_in_order():
    e = parse_expression("a*x[-1] + exp(b - y) / 2", ["a", "b"])
    order = [str(n) for n in e.walk()]
    assert order == [str(e), "a*x[-1]", "a", "x[-1]", "exp(b - y)/2", "exp(b - y)", "b - y", "b", "y", "2"]


def test_the_jacobian_of_a_long_sum():
    """Every coefficient and regressor of a 300-term sum (a deep tree): each derivative is its partner, or zero, and
    the whole Jacobian takes well under a second (it took minutes when every subtree was searched again)."""
    import time

    k = 300
    names = [f"b{i}" for i in range(k)]
    e = parse_expression(" + ".join(f"b{i}*x{i}[-1]" for i in range(k)), names)
    start = time.perf_counter()
    by_coef = [diff(e, Coef(n)) for n in names]
    by_name = [diff(e, Name(f"x{i}", -1)) for i in range(k)]
    elapsed = time.perf_counter() - start
    assert all(d == Name(f"x{i}", -1) for i, d in enumerate(by_coef))
    assert all(d == Coef(f"b{i}") for i, d in enumerate(by_name))
    assert diff(e, Name("x0", 0)) == parse_expression("0")
    assert elapsed < 20, f"{elapsed:.1f}s"


def test_unknown_functions_suggest_the_name_pymetrix_uses():
    with pytest.raises(ExpressionError, match=r"unknown function 'diff': did you mean d\(x\)"):
        parse_expression("diff(log(y))")
    with pytest.raises(ExpressionError, match=r"did you mean movavg\(\)\?"):
        parse_expression("movavgg(y, 4)")
    with pytest.raises(ExpressionError, match=r"unknown function 'foo' \(known: "):  # nothing close: the list only
        parse_expression("foo(y)")
