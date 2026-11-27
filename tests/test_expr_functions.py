"""The expression functions of the data's rows and of missing values: the calendar (year(), quarter(), month(), day(),
obsnum()), rounding (round, floor, ceil), log10, missing values (na, isna, fillna), the conditional iif and seeded
random draws (normrnd, unifrnd). Each is checked against NumPy, pandas or the decimal module, and through every path an
expression takes: the parser and its printing, the evaluator, the compiler, derivatives, models and the model writers."""

from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, Model, ModelSource
from pymetrix._random import generator, random_generator
from pymetrix.expr import ExpressionError, Name, diff, evaluate, parse_expression
from pymetrix.models.solvers import ModelError


@pytest.fixture
def quarterly() -> Dataset:
    rng = np.random.default_rng(3)
    n = 12
    y = rng.normal(size=n) * 10
    y[[1, 6]] = np.nan
    frame = pd.DataFrame({"x": rng.normal(size=n) * 10, "y": y},
                         index=pd.period_range("1999Q3", periods=n, freq="Q", name="time"))
    return Dataset.from_frame(frame)


def test_round_trip_through_str():
    for text in ["round(x)", "round(x, 2)", "round(x, -1)", "floor(x) + ceil(y)", "log10(x)", "isna(x)", "fillna(x, 0)",
                 "iif(x > 0, x, na)", "na", "year() + quarter()/10", "month() - day()", "obsnum()",
                 "normrnd(seed=1)", "normrnd(1, seed=2)", "normrnd(1, 3, seed=2)", "unifrnd(2, 5, seed=7)"]:
        e = parse_expression(text)
        assert parse_expression(str(e)) == e, text
    assert str(parse_expression("normrnd(0, 1, seed=4)")) == "normrnd(seed=4)"  # the defaults left out
    assert str(parse_expression("unifrnd(0, 2, seed=4)")) == "unifrnd(0, 2, seed=4)"


@pytest.mark.parametrize("text, message", [
    ("ceiling(x)", "did you mean ceil"), ("ifelse(x > 0, 1, 2)", "iif"), ("isnan(x)", "isna"), ("nan_to(x, 0)", "fillna"),
    ("rnorm()", "normrnd"), ("normrnd()", "needs seed="), ("normrnd(seed=1.5)", "whole number"),
    ("normrnd(0, 1, 2, seed=1)", "at most two"), ("abs(x, seed=1)", "does not take seed="), ("year(x)", "no arguments"),
    ("round(x, 0.5)", "whole number"), ("round(x, n)", "whole number"), ("iif(x, 1)", "3 arguments"),
    ("fillna(x)", "2 arguments"), ("round()", "1 or 2"),
    ("d(year())", "cannot be lagged"), ("obsnum()[-1]", "cannot be lagged"), ("movavg(month(), 3)", "cannot be lagged"),
    ("pch(normrnd(seed=1))", r"s = normrnd\(seed=1\)"), ("cmovavg(quarter(), 3)", "cannot be lagged"),
])
def test_errors_name_the_right_form(text, message):
    with pytest.raises(ExpressionError, match=message):
        parse_expression(text)


def test_calendar_of_periods():
    index = pd.period_range("1999Q3", periods=6, freq="Q", name="time")
    data = Dataset.from_frame(pd.DataFrame({"x": np.arange(6.0)}, index=index))
    starts = index.start_time
    for fn, expected in [("year", starts.year), ("quarter", starts.quarter), ("month", starts.month), ("day", starts.day)]:
        np.testing.assert_array_equal(data.eval(f"{fn}()").to_numpy(), np.asarray(expected, float))
    np.testing.assert_array_equal(data.eval("year()").to_numpy(), [1999, 1999, 2000, 2000, 2000, 2000])
    np.testing.assert_array_equal(data.eval("month()").to_numpy(), [7, 10, 1, 4, 7, 10])  # a quarter's first month
    np.testing.assert_array_equal(data.eval("obsnum()").to_numpy(), np.arange(1.0, 7.0))
    monthly = Dataset.from_frame(pd.DataFrame({"x": np.zeros(14)}, index=pd.period_range("2023-11", periods=14, freq="M", name="time")))
    np.testing.assert_array_equal(monthly.eval("month()").to_numpy(), [11, 12] + list(range(1, 13)))
    np.testing.assert_array_equal(monthly.eval("quarter()").to_numpy(), [4, 4, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    daily = Dataset.from_frame(pd.DataFrame({"x": np.zeros(5)}, index=pd.period_range("2024-02-27", periods=5, freq="D", name="time")))
    np.testing.assert_array_equal(daily.eval("day()").to_numpy(), [27, 28, 29, 1, 2])
    # conditions on the calendar: a dummy for the fourth quarter, a break after 2000
    np.testing.assert_array_equal(data.eval("quarter() == 4").to_numpy(), [0, 1, 0, 0, 0, 1])
    np.testing.assert_array_equal(data.eval("year() >= 2000").to_numpy(), [0, 0, 1, 1, 1, 1])


def test_obsnum_counts_within_each_entity_in_time_order(panel):
    numbers = panel.eval("obsnum()")
    frame = panel.frame.reset_index()
    expected = frame.groupby("country")["year"].rank(method="first").to_numpy()  # pandas' own count, time order
    np.testing.assert_array_equal(numbers.to_numpy(), expected)
    assert numbers.max() == 20 and numbers.min() == 1
    np.testing.assert_array_equal(panel.eval("year()").to_numpy(), frame["year"].dt.year.astype(float).to_numpy())


def test_obsnum_without_a_time_axis_is_the_row_number():
    data = Dataset.from_frame(pd.DataFrame({"id": ["c", "a", "b"], "x": [3.0, 1.0, 2.0]}), entity="id")  # a cross-section
    np.testing.assert_array_equal(data.eval("obsnum()").to_numpy(), [1.0, 2.0, 3.0])
    with pytest.raises(ExpressionError, match="needs a time axis"):
        data.eval("year()")


def test_rounding_against_numpy_and_decimal(quarterly):
    x = quarterly.frame["x"]
    np.testing.assert_array_equal(quarterly.eval("floor(x)").to_numpy(), np.floor(x.to_numpy()))
    np.testing.assert_array_equal(quarterly.eval("ceil(x)").to_numpy(), np.ceil(x.to_numpy()))
    np.testing.assert_allclose(quarterly.eval("log10(abs(x))").to_numpy(), np.log10(np.abs(x.to_numpy())), rtol=1e-15)

    def half_up(v: float, n: int) -> float:  # halves away from zero, on the number as written (its shortest form)
        return float(Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-n), rounding=ROUND_HALF_UP))

    for n in (0, 1, 2):
        got = quarterly.eval(f"round(x, {n})").to_numpy()
        np.testing.assert_allclose(got, [half_up(v, n) for v in x], rtol=1e-15, atol=0)
    halves = [2.5, -2.5, 0.5, 1.5, -0.5, 1234.0, 1250.0, 0.125]
    data = Dataset.from_frame(pd.DataFrame({"v": halves}, index=pd.period_range("2000", periods=8, freq="Y", name="time")))
    np.testing.assert_array_equal(data.eval("round(v)").to_numpy(), [3, -3, 1, 2, -1, 1234, 1250, 0])  # NumPy: 2, -2, 0, 2, -0
    np.testing.assert_array_equal(data.eval("round(v, -2)").to_numpy(), [0, -0, 0, 0, -0, 1200, 1300, 0])
    assert data.eval("round(v, 2)").iloc[-1] == 0.13  # 0.125 is exact in binary: the half goes up
    assert evaluate(parse_expression("round(-7.45, 1)"), lambda n, k: None) == -7.5
    for written, rounded in [("2.675", 2.68), ("1.005", 1.01), ("-1.005", -1.01), ("0.285", 0.29), ("1.015", 1.02)]:
        assert evaluate(parse_expression(f"round({written}, 2)"), lambda n, k: None) == rounded  # just below in binary
    many = np.random.default_rng(8).normal(scale=50, size=2000).round(3)  # many near-halves at 2 decimals
    for n in (0, 2):
        got = evaluate(parse_expression(f"round(v, {n})"), lambda name, lag: many)
        np.testing.assert_array_equal(got, [half_up(v, n) for v in many])


def test_missing_values(quarterly):
    y = quarterly.frame["y"]
    isna = quarterly.eval("isna(y)")
    np.testing.assert_array_equal(isna.to_numpy(), y.isna().astype(float).to_numpy())
    assert not isna.isna().any()  # never missing itself
    np.testing.assert_array_equal(quarterly.eval("fillna(y, 0)").to_numpy(), y.fillna(0).to_numpy())
    np.testing.assert_array_equal(quarterly.eval("fillna(y, x)").to_numpy(), y.fillna(quarterly.frame["x"]).to_numpy())
    np.testing.assert_array_equal(quarterly.eval("fillna(y, y[-1])").to_numpy(), y.fillna(y.shift(1)).to_numpy())
    assert quarterly.eval("na").isna().all() and quarterly.eval("x + na").isna().all()
    assert quarterly.eval("isna(na)").eq(1).all()


def test_iif(quarterly):
    x, y = quarterly.frame["x"], quarterly.frame["y"]
    got = quarterly.eval("iif(y > 0, x, -x)").to_numpy()
    expected = np.where(y.isna(), np.nan, np.where(y > 0, x, -x))  # missing where the condition is
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(quarterly.eval("iif(x > 0, x, na)").to_numpy(), x.where(x > 0).to_numpy())
    np.testing.assert_array_equal(quarterly.eval("iif(isna(y), 0, y)").to_numpy(), y.fillna(0).to_numpy())
    np.testing.assert_array_equal(quarterly.eval("iif(x, 1, 2)").to_numpy(), np.where(x != 0, 1.0, 2.0))  # nonzero is true
    assert evaluate(parse_expression("iif(1 > 2, 3, 4)"), lambda n, k: None) == 4.0


def test_random_draws_are_the_seeded_generators(quarterly):
    n = len(quarterly.frame)
    np.testing.assert_array_equal(quarterly.eval("normrnd(seed=5)").to_numpy(), generator(5).standard_normal(n))
    np.testing.assert_allclose(quarterly.eval("normrnd(2, 3, seed=5)").to_numpy(), 2 + 3 * generator(5).standard_normal(n), rtol=1e-15)
    np.testing.assert_allclose(quarterly.eval("unifrnd(2, 5, seed=5)").to_numpy(), 2 + 3 * generator(5).random(n), rtol=1e-15)
    np.testing.assert_allclose(quarterly.eval("unifrnd(2, 5, seed=5)").to_numpy(), np.random.default_rng(5).uniform(2, 5, n), rtol=1e-15)
    np.testing.assert_allclose(quarterly.eval("normrnd(2, 3, seed=5)").to_numpy(), np.random.default_rng(5).normal(2, 3, n), rtol=1e-15)
    a, b = quarterly.eval("normrnd(seed=5)"), quarterly.eval("normrnd(seed=5)")
    assert a.equals(b) and not a.equals(quarterly.eval("normrnd(seed=6)"))
    with random_generator("mt19937"):  # the package's choice of generator
        mt = quarterly.eval("normrnd(seed=5)").to_numpy()
    np.testing.assert_array_equal(mt, np.random.Generator(np.random.MT19937(5)).standard_normal(n))
    scaled = quarterly.eval("normrnd(0, abs(x), seed=5)").to_numpy()  # parameters may be series
    np.testing.assert_allclose(scaled, np.abs(quarterly.frame["x"].to_numpy()) * generator(5).standard_normal(n), rtol=1e-15)
    assert np.isnan(quarterly.eval("normrnd(0, -1, seed=5)")).all()  # a negative standard deviation, as normpdf gives


def test_generated_and_derived_series(quarterly):
    made = quarterly.with_series(e="normrnd(seed=1)", yr="year()", q4="quarter() == 4")
    np.testing.assert_array_equal(made.frame["e"].to_numpy(), generator(1).standard_normal(len(made.frame)))
    assert made.frame["yr"].iloc[0] == 1999 and made.frame["q4"].sum() == 3
    derived = quarterly.derive(filled="fillna(y, 0)", r="round(x, 1)")
    np.testing.assert_array_equal(derived.frame["filled"].to_numpy(), quarterly.frame["y"].fillna(0).to_numpy())
    changed = derived.with_series(y=quarterly.frame["y"].fillna(7.0))  # a derived series follows its input
    assert changed.frame["filled"].iloc[1] == 7.0
    with pytest.raises(Exception, match="draws random numbers"):  # draws are values, not a recipe
        quarterly.derive(e="normrnd(seed=1)")


def test_calendar_dummies_in_an_equation(ts):
    inline = Equation("y ~ x + (quarter() == 4) + (year() >= 2010)").fit(ts)
    index = ts.frame.index
    made = ts.with_series(q4=pd.Series((index.quarter == 4).astype(float), index=index),
                          late=pd.Series((index.year >= 2010).astype(float), index=index))
    stored = Equation("y ~ x + q4 + late").fit(made)
    np.testing.assert_allclose(np.asarray(inline.params.values), np.asarray(stored.params.values), rtol=1e-12)


def test_compiled_agrees_with_the_interpreter():
    from pymetrix.expr.compile import compile_exprs, compilable

    texts = ["round(a, 1) + floor(b) + ceil(c)", "log10(abs(a) + 1)", "isna(b) + fillna(b, 9)", "iif(a > 0, b, c)",
             "iif(b, a, na)", "fillna(na, a)"]
    exprs = [parse_expression(t) for t in texts]
    assert all(compilable(e) for e in exprs)
    assert not compilable(parse_expression("year() + a")) and not compilable(parse_expression("normrnd(seed=1)"))
    compiled = compile_exprs(exprs)
    cases = [(2.45, 1.5, -1.5), (-2.5, np.nan, 0.2), (0.0, 0.0, 3.7), (np.nan, 2.0, 1.0)]
    columns = [np.array(c) for c in zip(*cases)]
    order = [{"a": 0, "b": 1, "c": 2}[name] for name, _ in compiled.refs]
    with np.errstate(all="ignore"):
        arrays = compiled([columns[k] for k in order])
        for e, arr in zip(exprs, arrays):
            expected = evaluate(e, lambda name, lag: columns["abc".index(name)])
            np.testing.assert_array_equal(np.broadcast_to(arr, (len(cases),)), np.broadcast_to(expected, (len(cases),)), err_msg=str(e))
        for i, case in enumerate(cases):
            for e, s in zip(exprs, compiled([case[k] for k in order])):
                np.testing.assert_array_equal(s, evaluate(e, lambda name, lag, c=case: c["abc".index(name)]), err_msg=f"{e} at {case}")


def test_derivatives():
    e = parse_expression("log10(x)")
    point = {"x": 3.7}
    d = evaluate(diff(e, Name("x")), lambda n, k: point[n])
    h = 1e-6
    numeric = (np.log10(point["x"] + h) - np.log10(point["x"] - h)) / (2 * h)
    assert d == pytest.approx(numeric, rel=1e-8) and d == pytest.approx(1 / (3.7 * np.log(10)), rel=1e-15)
    for flat in ["round(x)", "floor(x)", "ceil(x)", "isna(x)", "round(x, 2)"]:  # steps: flat where they do not jump
        assert evaluate(diff(parse_expression(flat), Name("x")), lambda n, k: 1.3) == 0.0
    branch = diff(parse_expression("iif(z > 0, x^2, 3*x)"), Name("x"))
    assert evaluate(branch, lambda n, k: {"x": 2.0, "z": 1.0}[n]) == 4.0
    assert evaluate(branch, lambda n, k: {"x": 2.0, "z": -1.0}[n]) == 3.0
    filled = diff(parse_expression("fillna(x, 2*w)"), Name("w"))
    assert evaluate(filled, lambda n, k: {"x": np.nan, "w": 1.0}[n]) == 2.0
    assert evaluate(filled, lambda n, k: {"x": 1.0, "w": 1.0}[n]) == 0.0


def test_models_refuse_functions_of_the_data_rows():
    for text in ["y = 0.5*y[-1] + 0.1*year()", "y = 0.5*y[-1] + normrnd(seed=1)", "y = 0.5*y[-1] + obsnum()"]:
        with pytest.raises(ModelError, match="function of the data's rows"):
            Model([Equation(text)])


def test_models_use_rounding_and_choices():
    index = pd.period_range("2000Q1", periods=8, freq="Q", name="time")
    e = np.array([0.0, 1.26, -3.45, 0.5, 0.25, 0.0, 2.0, 0.0])
    data = Dataset.from_frame(pd.DataFrame({"x": np.zeros(8), "z": np.zeros(8), "e": e}, index=index))
    model = Model([Equation("x = 0.5*x[-1] + e"), Equation("z = iif(x > 0, round(x, 1), floor(x)) + log10(2)")])
    solved = model.solve(data, ("2000Q2", "2001Q4")).frame
    x = np.zeros(8)
    for t in range(1, 8):
        x[t] = 0.5 * x[t - 1] + e[t]
    np.testing.assert_allclose(solved["x"].to_numpy(), x, atol=1e-12)
    half_up = np.sign(x) * np.floor(np.abs(x) * 10 + 0.5) / 10
    expected = np.where(x > 0, half_up, np.floor(x)) + np.log10(2)
    np.testing.assert_allclose(solved["z"].to_numpy()[1:], expected[1:], atol=1e-12)


def test_writers():
    text = "shocks\n    e = 0.1\nequations\n    x = 0.5*x[-1] + e\n    z = log10(2 + x)\n"
    dynare = ModelSource(text).export("dynare")
    assert "log10(2 + x)" in dynare.text and dynare.checked
    assert "log10(2 + x)" in ModelSource(text).export("iris").text
    for bad in ["round(x)", "floor(x)", "iif(x > 0, x, 0)", "fillna(x, 0)"]:
        with pytest.raises(ModelError, match="no equivalent in Dynare"):
            ModelSource(text.replace("log10(2 + x)", bad)).export("dynare")


def test_notation():
    from pymetrix.notation import expression

    assert expression(parse_expression("iif(x > 0, x, na)")) == "iif(xₜ > 0, xₜ, NA)"
    assert "\\text{NA}" in expression(parse_expression("fillna(na, x)"), "latex")
    assert expression(parse_expression("round(x, 2)")) == "round(xₜ, 2)"


def test_model_text_knows_the_functions():
    from pymetrix.models.source import function_help

    for fn in ["log10", "floor", "ceil", "round", "isna", "fillna", "iif", "year", "obsnum", "normrnd", "unifrnd"]:
        assert function_help(fn), fn
    check = ModelSource("shocks\n    e = 0.1\nequations\n    x = 0.5*x[-1] + e\n    z = round(x, 1) + iif(x > 0, 1, 0)\n").check()
    assert not check.diagnostics, check.diagnostics
