"""Seasonal (period-of-year) dummies in formulas: ``seasonal()``, and ``C(quarter)`` / ``C(month)`` with the quarter
or month taken from the time axis.

Checked against: the same regression with 0/1 dummies built here by hand from the periods' quarter or month (numpy
comparisons), the coefficients and standard errors equal to 1e-12; with an intercept the first season is left out,
without one every season has a dummy.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation
from pymetrix.commands import FitEquation, Session
from pymetrix.errors import PyMetrixError
from pymetrix.expr import ExpressionError
from pymetrix.ui.formula_check import check_formula


def _data(freq: str, n: int, start: str, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    index = pd.period_range(start, periods=n, freq=freq, name="time")
    season = (index.quarter if freq.startswith("Q") else index.month).to_numpy()
    x = rng.normal(size=n)
    effect = {s: 0.3 * s for s in range(1, 13)}
    y = 1 + 0.5 * x + np.array([effect[s] for s in season]) + rng.normal(scale=0.2, size=n)
    return Dataset.from_frame(pd.DataFrame({"y": y, "x": x}, index=index))


def _by_hand(data: Dataset, field: str, intercept: bool, prefix: str):
    """The dummies built by hand from the periods, and the regression on them written with plain names."""
    periods = data.frame.index
    season = np.asarray(getattr(periods, field))
    cats = sorted(set(season))[1:] if intercept else sorted(set(season))
    frame = data.frame.copy()
    for c in cats:
        frame[f"{prefix}{c}"] = (season == c).astype(float)
    rhs = " + ".join(["x"] + [f"{prefix}{c}" for c in cats]) + ("" if intercept else " - 1")
    return Equation(f"y ~ {rhs}").fit(Dataset.from_frame(frame)), cats


CASES = [("Q", 40, "2005Q2", "quarter", "seasonal[Q{}]", "C(quarter)[{}]"),
         ("M", 72, "2010-05", "month", "seasonal[M{}]", "C(month)[{}]")]


@pytest.mark.parametrize("freq,n,start,field,seasonal_name,c_name", CASES)
@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("term", ["seasonal()", "C"])
def test_equal_to_dummies_built_by_hand(freq, n, start, field, seasonal_name, c_name, intercept, term):
    data = _data(freq, n, start)  # starting mid-year: the dummies follow the periods, not the row order
    written = "seasonal()" if term == "seasonal()" else f"C({field})"
    fit = Equation(f"y ~ x + {written}" + ("" if intercept else " - 1")).fit(data)
    ref, cats = _by_hand(data, field, intercept, "s")
    name = seasonal_name if term == "seasonal()" else c_name
    expected = (["const"] if intercept else []) + ["x"] + [name.format(c) for c in cats]
    assert list(fit.params.names) == expected
    assert len(cats) == (4 if freq == "Q" else 12) - intercept  # one per season less one with an intercept
    np.testing.assert_allclose(fit.params.values, ref.params.values, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fit.params.se, ref.params.se, rtol=0, atol=1e-12)
    assert fit.stats["nobs"] == n


def test_dummies_align_with_the_period_index():
    """A sample that starts in Q3 and has a gap: each row's dummy is its own quarter's."""
    data = _data("Q", 40, "2001Q3")
    frame = data.frame.drop(data.frame.index[[3, 4, 9]])
    gappy = Dataset.from_frame(frame)
    seasons = gappy.values_of("seasonal()")
    assert (seasons.to_numpy() == frame.index.quarter.to_numpy()).all()
    fit = Equation("y ~ x + seasonal()").fit(gappy)
    ref, _ = _by_hand(gappy, "quarter", True, "s")
    np.testing.assert_allclose(fit.params.values, ref.params.values, atol=1e-12)
    rows = Equation("y ~ x + seasonal()").fit(gappy, span=("2003Q1", "2008Q4"))
    assert rows.stats["nobs"] == 23  # 24 quarters less 2003Q4, dropped above


def test_fiscal_quarters_and_a_column_named_quarter():
    data = _data("Q-JUN", 24, "2010Q1")  # fiscal years ending in June: quarter 1 is July to September
    assert (data.values_of("quarter").to_numpy() == data.frame.index.quarter.to_numpy()).all()
    own = Dataset.from_frame(data.frame.assign(quarter=np.arange(24) % 2))  # a series named quarter comes first
    fit = Equation("y ~ x + C(quarter)").fit(own)
    assert [n for n in fit.params.names if n.startswith("C(")] == ["C(quarter)[1]"]


def test_errors():
    annual = _data("Y", 20, "2000")
    with pytest.raises(PyMetrixError, match="quarterly or monthly"):
        Equation("y ~ x + seasonal()").fit(annual)
    with pytest.raises(PyMetrixError, match="no quarters"):
        Equation("y ~ x + C(quarter)").fit(annual)
    with pytest.raises(PyMetrixError, match="no months"):
        Equation("y ~ x + C(month)").fit(_data("Q", 20, "2000Q1"))
    with pytest.raises(ExpressionError, match="no arguments"):
        Equation("y ~ x + seasonal(x)")
    with pytest.raises(ExpressionError, match="terms on their own"):
        Equation("y ~ x + x * seasonal()")
    with pytest.raises(ExpressionError, match="linear-form"):
        Equation("y ~ a + b * x + seasonal()", params=["a", "b"])


def test_wald_on_the_seasonal_dummies():
    """The dummies' names work in the Wald test: no seasonality is all of them zero, an F test equal to the one from
    the regression with and without them."""
    data = _data("Q", 60, "2000Q1")
    fit = Equation("y ~ x + seasonal()").fit(data)
    wald = fit.test("wald", "seasonal[Q2] = 0, seasonal[Q3] = 0, `seasonal[Q4]` = 0")
    short = Equation("y ~ x").fit(data)
    F = ((short.stats["ssr"] - fit.stats["ssr"]) / 3) / (fit.stats["ssr"] / fit.stats["df_resid"])
    assert wald.details["F"] == pytest.approx(F, rel=1e-10)


def test_formula_check_accepts_them():
    data = _data("Q", 20, "2000Q1")
    names = [*data.names, *[n for n in data.frame.index.names if n], *data.calendar_names()]
    assert data.calendar_names() == ["quarter"] and _data("M", 24, "2000-01").calendar_names() == ["quarter", "month"]
    for text in ("y ~ x + seasonal()", "y ~ x + C(quarter)", "y ~ x + seasonal() - 1"):
        found = check_formula(text, "equation", names, equation_class=True)
        assert not found.message, (text, found.message)
    assert "quarter" in check_formula("y ~ x + C(quarter)", "equation", data.names).message  # without the time axis's names


def test_dialog_check_api_and_script_replay():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from pymetrix.ui.qt import command_dialog

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    data = _data("M", 48, "2015-01")
    s = Session()
    s.objects["data"] = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dialog = command_dialog(FitEquation, s)
    dialog.editors["formula"].set("y ~ x + seasonal() + C(quarter)")
    dialog.check_formulas()
    assert not dialog.formula_checks["formula"].message  # neither an error nor "not in the data"
    dialog.editors["formula"].set("y ~ x + seasonal()")
    dialog.check_formulas()
    assert not dialog.formula_checks["formula"].message
    dialog.close()
    s.run(FitEquation(formula="y ~ x + seasonal()", name="fit"))
    by_api = Equation("y ~ x + seasonal()").fit(data)
    np.testing.assert_array_equal(s["fit"].params.values, by_api.params.values)
    assert list(s["fit"].params.names) == list(by_api.params.names) and "seasonal[M12]" in by_api.params.names
    ns = dict(s.objects)
    exec(s.script(), ns)
    np.testing.assert_array_equal(ns["fit"].params.values, by_api.params.values)
