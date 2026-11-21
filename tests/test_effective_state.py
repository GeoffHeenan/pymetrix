"""Effective state in every result (plan item 3): the rows actually used, after missing values, lags and the
library's own presample; the backend that did the numbers; warnings from every estimator."""

import warnings

import numpy as np
import pandas as pd
import pytest

import pymetrix
from pymetrix import ARDL, VAR, VECM, Dataset, Equation, ExponentialSmoothing, MarkovSwitching, System
from pymetrix._capture import effective_sample


@pytest.fixture(scope="module")
def data():
    return Dataset.example().with_series(lc="log(c)", ly="log(y)", g4="100*dlog(y)")


def test_rows_dropped_are_listed_with_their_reason():
    idx = pd.period_range("2000Q1", periods=40, freq="Q")
    rng = np.random.default_rng(0)
    y = rng.normal(size=40)
    y[10] = np.nan
    ds = Dataset.from_frame(pd.DataFrame({"y": y, "x": rng.normal(size=40)}, index=idx))
    p = Equation("y ~ x + y[-1]").fit(ds, span=("2000Q1", "2008Q4")).provenance
    assert (p.sample, p.nobs, p.dropped) == (("2000Q2", "2008Q4"), 33, 3)
    assert p.rows == [["2000Q2", "2002Q2"], ["2003Q1", "2008Q4"]]  # the gap is visible
    assert p.dropped_rows["by_reason"] == {"lag": 2, "missing": 1}
    assert ["2002Q3", "missing"] in p.dropped_rows["rows"] and ["2002Q4", "lag"] in p.dropped_rows["rows"]
    assert p.backend == {"name": "pymetrix", "version": pymetrix.__version__, "wraps": "pymetrix.OLS"}


def test_presample_taken_by_the_library_is_counted(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ardl = ARDL("lc ~ ly", lags=2, order=1).fit(data)
        vecm = VECM(["lc", "ly"], lags=2, rank=1).fit(data)
        markov = MarkovSwitching("g4 ~ 1", ar=1).fit(data)
    assert ardl.provenance.nobs == len(ardl.resid) and ardl.provenance.presample == ("2000Q1", "2000Q2")
    assert vecm.provenance.nobs == int(vecm.stats["nobs"]) == 97  # was 100: statsmodels takes lags + 1 rows
    assert vecm.provenance.presample == ("2000Q1", "2000Q2", "2000Q3") and vecm.provenance.sample[0] == "2000Q4"
    assert markov.provenance.presample == ("2000Q2",)  # the AR lag; 2000Q1 is missing (a growth rate)
    assert markov.provenance.dropped_rows["by_reason"] == {"missing": 1, "presample": 1}


def test_first_differences_report_the_libraries_observations():
    pytest.importorskip("linearmodels")
    rng = np.random.default_rng(1)
    rows = [{"id": e, "year": 2000 + t, "x": rng.normal(), "y": rng.normal()} for e in range(10) for t in range(8)]
    d = Dataset.from_frame(pd.DataFrame(rows), entity="id", time="year", freq="Y")
    fd = Equation("y ~ x").fit(d, method=pymetrix.FirstDifference())
    assert fd.provenance.nobs == int(fd.stats["nobs"]) == 70  # was 80
    assert fd.provenance.rows["0"] == [["2001", "2007"]]
    between = Equation("y ~ x").fit(d, method=pymetrix.Between())
    assert between.provenance.nobs == 10 and between.provenance.rows is None  # one row per entity


def test_exponential_smoothing_counts_the_trimmed_ends(data):
    gappy = data.with_series(z="c").edit({"z": {"2000Q1": float("nan"), "2000Q2": float("nan")}})
    p = ExponentialSmoothing("z").fit(gappy).provenance
    assert p.nobs == 98 and p.dropped_rows["by_reason"] == {"missing": 2}


def test_var_and_system_record_their_rows(data):
    var = VAR(["lc", "ly"], lags=2).fit(data, span=("2001Q1", "2020Q4"))
    assert var.provenance.nobs == 80 and var.provenance.dropped == 0  # lags reach before the span: nothing lost
    whole = VAR(["lc", "ly"], lags=2).fit(data)
    assert whole.provenance.dropped_rows["by_reason"] == {"lag": 2}
    system = System([Equation("lc ~ ly"), Equation("ly ~ lc[-1]")]).fit(data)
    assert system.provenance.dropped == 1 and system.provenance.backend["name"] == "pymetrix"


def test_state_space_counts_periods_with_observations(data):
    uc = pymetrix.UnobservedComponents("ly", cycle=0)
    gappy = data.edit({"ly": {"2010Q1": float("nan")}})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = uc.fit(gappy)
    p = fit.provenance
    assert p.nobs == 99 and p.rows == [["2000Q1", "2009Q4"], ["2010Q2", "2024Q4"]]
    assert p.backend["name"] in ("statsmodels", "pymetrix")


def test_tests_on_data_record_the_observations_tested_and_gaps(data):
    gappy = data.with_series(z="y").edit({"z": {"2010Q1": float("nan")}})
    p = gappy.test("adf", "z").provenance
    assert p.nobs == 99 and p.dropped == 1
    assert any("gaps inside the sample" in m for m in p.messages)
    assert p.backend["name"] == "statsmodels" and p.backend["wraps"] == "statsmodels.tsa.stattools.adfuller"


def test_warnings_are_kept_for_vars_systems_and_state_space(data, monkeypatch):
    import pymetrix.var as var_module

    original = var_module._estimate if hasattr(var_module, "_estimate") else None
    calls = []

    def warn(*args, **kwargs):
        warnings.warn("a warning raised while estimating", RuntimeWarning)
        calls.append(1)
        return original(*args, **kwargs)

    if original is None:
        pytest.skip("VAR has no _estimate to instrument")
    monkeypatch.setattr(var_module, "_estimate", warn)
    with pytest.warns(RuntimeWarning):
        fit = VAR(["lc", "ly"], lags=1).fit(data)
    assert calls and "RuntimeWarning: a warning raised while estimating" in fit.provenance.warnings


def test_effective_sample_without_a_time_axis():
    cross = Dataset.from_frame(pd.DataFrame({"id": range(5), "y": [1.0, 2, np.nan, 4, 5]}), entity="id")
    out = effective_sample(cross, cross.frame.index[[0, 1, 3, 4]], variables=["y"])
    assert out["rows"] is None and out["dropped"] == 1 and out["dropped_rows"]["by_reason"] == {"missing": 1}


def test_long_lists_of_dropped_rows_are_capped():
    idx = pd.period_range("1900-01", periods=200, freq="M")
    y = np.where(np.arange(200) % 2 == 0, np.nan, 1.0)
    ds = Dataset.from_frame(pd.DataFrame({"y": y}, index=idx))
    out = effective_sample(ds, idx[1::2], variables=["y"])
    assert out["dropped"] == 100 and len(out["dropped_rows"]["rows"]) == 50 and out["dropped_rows"]["more"] == 50
