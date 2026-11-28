"""pymetrix.ols: least squares on arrays passed directly, checked against Equation.fit on the same data."""

import dataclasses
import time

import numpy as np
import pandas as pd
import pytest

import pymetrix
from pymetrix import OLS, Cluster, DataError, Dataset, Equation, FittedEquation, SpecificationError, as_dataset, ols

TOL = 1e-12


def _close(a, b, tol=TOL):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape
    assert np.allclose(a, b, rtol=tol, atol=tol, equal_nan=True), np.nanmax(np.abs(a - b))


def _series(rng, n=40, k=3, time_series=True):
    index = pd.period_range("2000Q1", periods=n, freq="Q", name="time") if time_series else pd.RangeIndex(n, name="obs")
    X = pd.DataFrame(rng.normal(size=(n, k)), columns=[f"x{j + 1}" for j in range(k)], index=index)
    y = (0.5 + X.to_numpy() @ np.linspace(1, 0.2, k) + rng.normal(size=n)).astype(float)
    y = pd.Series(np.cumsum(y) * 0.1 + y, index=index, name="y")  # some serial correlation for the tests
    return y, X


def _dataset(y, X, *extra, time_series=True):
    frame = pd.concat([y, X, *extra], axis=1)
    if time_series:
        return as_dataset(frame)
    return Dataset.from_frame(frame.reset_index(), entity="obs")


def _same(direct, formula, *, tests=True):
    """Every number and every test of the two fits agree, and their provenance but for the documented difference."""
    assert isinstance(direct, FittedEquation) and type(direct.method) is type(formula.method) and direct.method == formula.method
    assert direct.equation.text == formula.equation.text
    assert list(direct.params.names) == list(formula.params.names)
    _close(direct.params.values, formula.params.values)
    _close(direct.params.cov, formula.params.cov)
    _close(direct.params.se, formula.params.se)
    assert set(direct.stats) == set(formula.stats)
    for k in formula.stats:
        _close(direct.stats[k], formula.stats[k])
    _close(direct.resid, formula.resid)
    _close(direct.fitted, formula.fitted)
    assert list(direct.resid.index) == list(formula.resid.index)
    _close(direct.design, formula.design)
    assert list(direct.design.columns) == list(formula.design.columns)
    assert direct.weights == formula.weights
    for field in dataclasses.fields(direct.provenance):
        a, b = getattr(direct.provenance, field.name), getattr(formula.provenance, field.name)
        if field.name == "created":
            continue
        if field.name == "vintage_request":
            assert a == {"policy": "direct"}
            continue
        assert a == b, (field.name, a, b)
    if tests:
        assert direct.tests() == formula.tests()
        for name, opts in _runnable(formula):
            r1, r2 = direct.test(name, **opts), formula.test(name, **opts)
            _close(r1.statistic, r2.statistic)
            if r2.p_value is not None:
                _close(r1.p_value, r2.p_value)
        d1, d2 = direct.diagnostics(), formula.diagnostics()
        assert [r[0] for r in d1.rows] == [r[0] for r in d2.rows]
        _close([r[1:3] for r in d1.rows], [r[1:3] for r in d2.rows])
        assert str(direct.summary()) == str(formula.summary())


def _runnable(fit):
    """Each test that applies, with the settings it needs; those that refuse this fit (weighted, cross-section) skipped."""
    needed = {"wald": {"restriction": f"{fit.params.names[-1]} = 0"},
              "chow": {"at": str(fit.resid.index[len(fit.resid) // 2])} if isinstance(fit.resid.index, pd.PeriodIndex) else None}
    out = []
    for name in fit.tests():
        opts = needed.get(name, {})
        if opts is None:
            continue
        try:
            fit.test(name, **opts)
        except (SpecificationError, NotImplementedError, TypeError):
            continue
        out.append((name, opts))
    assert out
    return out


@pytest.mark.parametrize("cov, opts", [
    ("classic", {}), ("white", {}), ("hac", {}), ("hac", {"hac_lags": 2}), ("hac", {"hac_lags": 3, "hac_correction": False}),
])
def test_time_series_matches_the_formula_path(cov, opts):
    rng = np.random.default_rng(1)
    y, X = _series(rng)
    y.iloc[3] = np.nan
    X.iloc[7, 1] = np.nan  # rows with missing values are dropped, and the provenance says why
    direct = ols(y, X, cov=cov, **opts)
    formula = Equation("y ~ x1 + x2 + x3").fit(_dataset(y, X), method=OLS(cov=cov, **opts))
    _same(direct, formula)
    assert direct.provenance.dropped == 2 and direct.provenance.dropped_rows["by_reason"] == {"missing": 2}
    assert direct.provenance.rows == [["2000Q1", "2000Q3"], ["2001Q1", "2001Q3"], ["2002Q1", "2009Q4"]]
    assert direct.provenance.backend["wraps"] == "pymetrix.OLS" and direct.provenance.data_fingerprint


def test_no_constant_weights_and_unsorted_rows():
    rng = np.random.default_rng(2)
    y, X = _series(rng, k=2)
    w = pd.Series(rng.uniform(0.5, 2.0, size=len(y)), index=y.index, name="w")
    formula_data = _dataset(y, X, w)
    _same(ols(y, X, constant=False), Equation("y ~ x1 + x2 - 1").fit(formula_data))
    _same(ols(y, X, weights=w), Equation("y ~ x1 + x2").fit(formula_data, weights="w"))
    y2 = y.copy()
    y2.iloc[5] = np.nan
    w2 = w.copy()
    w2.iloc[9] = np.nan
    _same(ols(y2, X, weights=w2, cov="white"), Equation("y ~ x1 + x2").fit(_dataset(y2, X, w2), weights="w", method=OLS(cov="white")))
    order = rng.permutation(len(y))  # rows come in any order: they are put in time order, as a dataset puts them
    _same(ols(y.iloc[order], X.iloc[order]), Equation("y ~ x1 + x2").fit(formula_data))


def test_cross_section_with_dummies_and_clusters():
    rng = np.random.default_rng(3)
    n = 120
    y, X = _series(rng, n=n, k=2, time_series=False)
    region = pd.Series(rng.choice(["north", "south", "west"], size=n), index=y.index, name="region")
    firm = pd.Series(rng.integers(0, 15, size=n), index=y.index, name="firm")
    dummies = pd.get_dummies(region, prefix="r", drop_first=True, dtype=float)
    Xd = pd.concat([X, dummies], axis=1)
    data = _dataset(y, Xd, region, firm, time_series=False)
    text = "y ~ x1 + x2 + r_south + r_west"
    _same(ols(y, Xd), Equation(text).fit(data))
    _same(ols(y, Xd, cov=Cluster("firm"), clusters=firm), Equation(text).fit(data, method=OLS(cov=Cluster("firm"))))
    _same(ols(y, Xd, cov="cluster", clusters=firm), Equation(text).fit(data, method=OLS(cov=Cluster("firm"))))
    both = pd.concat([firm, region], axis=1)
    _same(ols(y, Xd, cov=Cluster("firm", "region"), clusters=both),
          Equation(text).fit(data, method=OLS(cov=Cluster("firm", "region"))))
    # the dummies are the formula's C(region): the same estimates, under the formula's names
    categorical = Equation("y ~ x1 + x2 + C(region)").fit(data)
    _close(ols(y, Xd).params.values, categorical.params.values)
    _close(ols(y, Xd).params.se, categorical.params.se)


def test_panel_index_level_clusters():
    rng = np.random.default_rng(4)
    entities, periods = ["a", "b", "c", "d", "e"], pd.period_range("2001", periods=12, freq="Y")
    index = pd.MultiIndex.from_product([entities, periods], names=["country", "year"])
    X = pd.DataFrame(rng.normal(size=(len(index), 2)), columns=["x1", "x2"], index=index)
    y = pd.Series(X.sum(axis=1).to_numpy() + rng.normal(size=len(index)), index=index, name="y")
    data = Dataset.from_frame(pd.concat([y, X], axis=1).reset_index(), entity="country", time="year")
    _same(ols(y, X, cov=Cluster("country")), Equation("y ~ x1 + x2").fit(data, method=OLS(cov=Cluster("country"))), tests=False)
    _same(ols(y, X, cov=Cluster("country", "year")), Equation("y ~ x1 + x2").fit(data, method=OLS(cov=Cluster("country", "year"))), tests=False)


def test_numpy_inputs_and_names():
    rng = np.random.default_rng(5)
    Xa = rng.normal(size=(30, 2))
    ya = Xa @ [1.0, -1.0] + rng.normal(size=30)
    frame = pd.DataFrame({"y": ya, "x1": Xa[:, 0], "x2": Xa[:, 1], "obs": range(30)})
    formula = Equation("y ~ x1 + x2").fit(Dataset.from_frame(frame, entity="obs"))
    direct = ols(ya, Xa)
    assert direct.equation.text == "y ~ x1 + x2" and direct.provenance.sample is None
    _close(direct.params.values, formula.params.values)
    _close(direct.params.cov, formula.params.cov)
    assert direct.provenance.data_fingerprint == formula.provenance.data_fingerprint
    named = ols(pd.Series(ya, name="growth"), Xa, names=["capital", "labour"])
    assert named.equation.text == "growth ~ capital + labour" and list(named.params.names) == ["const", "capital", "labour"]
    _close(named.params.values, formula.params.values)
    mapping = ols(ya, {"capital": Xa[:, 0], "labour": Xa[:, 1]})
    _close(mapping.params.values, formula.params.values)
    structured = np.zeros(30, dtype=[("capital", float), ("labour", float)])
    structured["capital"], structured["labour"] = Xa[:, 0], Xa[:, 1]
    _close(ols(ya, structured).params.values, formula.params.values)
    one = ols(ya, Xa[:, 0])  # one regressor as a 1-d array
    assert list(one.params.names) == ["const", "x1"]


def test_dates_with_a_frequency_are_periods_and_forecasts_work():
    rng = np.random.default_rng(6)
    y, X = _series(rng, k=2)
    dated_y, dated_X = y.copy(), X.copy()
    dated_y.index = dated_X.index = pd.date_range("2000-01-01", periods=len(y), freq="QS")
    direct = ols(dated_y, dated_X)
    formula = Equation("y ~ x1 + x2").fit(_dataset(y, X))
    _same(direct, formula, tests=False)
    data = _dataset(y, X)  # a fit's equation is the formula of its names: it forecasts on any data with those series
    a, b = direct.forecast(data, ("2007Q1", "2009Q4")), formula.forecast(data, ("2007Q1", "2009Q4"))
    _close(a.frame.to_numpy(float), b.frame.to_numpy(float))
    _close(direct.predict(data), formula.predict(data))


def test_refusals_say_what_to_do():
    rng = np.random.default_rng(7)
    y, X = _series(rng, k=2)
    with pytest.raises(DataError, match="same index"):
        ols(y, X.set_axis(X.index + 1))
    with pytest.raises(SpecificationError, match="constant=False"):
        ols(y, X.assign(const=1.0))
    with pytest.raises(SpecificationError, match="series name"):
        ols(y, X.rename(columns={"x1": "x 1"}))
    with pytest.raises(SpecificationError, match="clash|series name|cannot be written"):
        ols(y, X.rename(columns={"x1": "and"}))
    with pytest.raises(SpecificationError, match="HAC"):
        ols(y.to_numpy(), X.to_numpy(), cov="hac")
    with pytest.raises(SpecificationError, match="Cluster"):
        ols(y, X, clusters=y.index.year)
    with pytest.raises(SpecificationError, match="missing values"):
        ols(y.where(y > y.min()), X, missing="raise")
    with pytest.raises(SpecificationError, match="collinear"):
        ols(y, X.assign(x3=X["x1"] * 2))
    with pytest.raises(SpecificationError, match="unknown covariance"):
        ols(y, X, cov="hc4")
    with pytest.raises(SpecificationError, match="both"):
        ols(y.rename("x1"), X)
    const_only = ols(y, X.iloc[:, :0])  # no regressors but the constant: y ~ 1
    assert const_only.equation.text == "y ~ 1"
    _close(const_only.params.values, [y.mean()])


def test_a_session_records_the_call_as_code():
    from pymetrix.commands import Session

    rng = np.random.default_rng(8)
    y, X = _series(rng, k=2)
    session = Session()
    session.objects["y"], session.objects["X"] = y, X
    session.run_code("from pymetrix import ols\nfit = ols(y, X, cov='white')")
    assert "fit = ols(y, X, cov='white')" in session.script()
    namespace = {"y": y, "X": X}
    exec(session.script(), namespace)
    _close(namespace["fit"].params.se, session["fit"].params.se)
    assert pymetrix.ols is ols


def _formula_loop(problems):
    for y, X in problems:
        Equation("y ~ " + " + ".join(X.columns)).fit(as_dataset(pd.concat([y, X], axis=1)))


def _direct_loop(problems):
    for y, X in problems:
        ols(y, X)


def test_many_small_regressions_are_much_faster_than_formulas():
    """The benchmark in docs/guide.md, small: 40 observations and 3 to 6 regressors. The direct path is recorded at
    about 4x faster; the test asks for 2x, the better of three interleaved runs of each, so a busy machine cannot fail it."""
    rng = np.random.default_rng(9)
    problems = [_series(rng, k=3 + i % 4) for i in range(60)]
    best = {"formula": np.inf, "direct": np.inf}
    for _ in range(3):
        for name, loop in (("formula", _formula_loop), ("direct", _direct_loop)):
            start = time.perf_counter()
            loop(problems)
            best[name] = min(best[name], time.perf_counter() - start)
    assert best["direct"] * 2 < best["formula"], best
