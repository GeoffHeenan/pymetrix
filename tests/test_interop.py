"""pymetrix with NumPy, pandas and statsmodels: out to ordinary objects on pymetrix's sample rules, back in with the
same checks as pymetrix's own data, and statsmodels results as ordinary pymetrix results."""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from pymetrix import OLS, DataError, Dataset, Equation, Sample


@pytest.fixture
def data():
    return Dataset.example("sample_macro")


def test_to_frame_keeps_names_index_sample_and_says_how_rows_were_chosen(data):
    frame = data.to_frame(["c", "y"])
    assert list(frame.columns) == ["c", "y"] and frame.index.equals(data.frame.index)
    assert frame.attrs["pymetrix"]["sample_source"] == "all observations"
    narrowed = data.with_sample(Sample(("2010Q1", "2014Q4")))
    part = narrowed.to_frame("c, y")  # the active sample, as commands use it
    assert str(part.index[0]) == "2010Q1" and len(part) == 20
    assert part.attrs["pymetrix"]["sample_source"] == "active sample of the dataset"
    assert len(narrowed.to_frame("c", sample="all")) == len(data.frame)
    assert np.array_equal(narrowed.to_numpy("c, y"), part.to_numpy())
    gappy = data.add_series("g2", data["g"].where(data["g"] > data["g"].median()))
    assert gappy.to_frame("c, g2", missing="keep").isna().any().any()
    dropped = gappy.to_frame("c, g2", missing="drop")
    assert not dropped.isna().any().any() and dropped.attrs["pymetrix"]["dropped"] == len(data.frame) - len(dropped)
    with pytest.raises(DataError, match="missing values"):
        gappy.to_frame("c, g2", missing="raise")
    with pytest.raises(DataError, match="no series nope"):
        data.to_frame("nope")


def test_design_gives_labelled_y_and_x_on_pymetrix_rows(data):
    y, X = data.design("c", ["y", "c[-1]"], sample=("2005Q1", None))
    assert y.name == "c" and list(X.columns) == ["const", "y", "c[-1]"] and (X["const"] == 1).all()
    assert str(X.index[0]) == "2005Q1" and y.index.equals(X.index)
    assert X.attrs["pymetrix"]["sample_source"] == "explicit" and X.attrs["pymetrix"]["equation"] == "c ~ y + c[-1]"
    native = Equation("c ~ y + c[-1]").fit(data, span=("2005Q1", None)).design
    assert np.allclose(X.to_numpy(), native.to_numpy())
    _, no_constant = data.design("log(c)", ["log(y)"], add_constant=False)
    assert list(no_constant.columns) == ["log(y)"]


def test_another_frequency_is_refused_not_turned_into_missing_values(data):
    monthly = pd.Series(np.arange(300.0), index=pd.period_range("2000-01", periods=300, freq="M"))
    with pytest.raises(DataError, match="is M, but this dataset is Q"):
        data.add_series("m", monthly)
    with pytest.raises(DataError, match="values for 100 rows"):
        data.add_series("short", np.ones(5))


def test_add_series_refuses_an_existing_name_unless_asked_and_records_its_source(data):
    with pytest.raises(DataError, match="already has a series c"):
        data.add_series("c", data["c"] * 2)
    replaced = data.add_series("c", data["c"] * 2, overwrite=True)
    assert np.allclose(replaced["c"], data["c"] * 2) and replaced.variables["c"].source == "user:pandas"
    added = data.add_series("noise", np.zeros(len(data.frame)), units="index")
    noise = added.variables["noise"]
    assert noise.source == "user:numpy" and noise.retrieved and noise.units == "index"
    logged = data.apply_np("lc", np.log, "c", source_expr="np.log(c)")
    assert np.allclose(logged["lc"], np.log(data["c"])) and logged.variables["lc"].derived_from == "np.log(c)"
    growth = data.apply_pd("gy", lambda s: s.pct_change(), "y")
    assert growth.variables["gy"].derived_from.startswith("user code:") and np.isnan(growth["gy"].iloc[0])
    each = data.transform_each(["c", "y"], np.log, prefix="l", source_expr="np.log({s})")
    assert each.variables["lc"].derived_from == "np.log(c)" and each.variables["ly"].derived_from == "np.log(y)"


def test_from_frame_keeps_the_index_frequency_and_records_a_source():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=pd.period_range("2020-01", periods=3, freq="M", name="month"))
    data = Dataset.from_frame(frame, source="user:pandas")
    assert data.frame.index.freqstr.startswith("M") and data.time_axis.name == "month"
    assert data.variables["x"].source == "user:pandas"
    assert Dataset.from_frame(frame).variables["x"].source == ""  # no source unless given


def test_a_statsmodels_fit_becomes_a_pymetrix_result(data):
    sm = pytest.importorskip("statsmodels.api")
    from pymetrix.estimation import FittedEquation

    y, X = data.design("c", ["y", "c[-1]"])
    res = sm.OLS(y, X).fit()
    fit = FittedEquation.from_statsmodels(res, data, command="sm.OLS(y, X).fit()")
    native = Equation("c ~ y + c[-1]").fit(data)
    assert fit.params.names == native.params.names
    np.testing.assert_allclose(fit.params.values, native.params.values, rtol=1e-10)
    np.testing.assert_allclose(fit.params.se, res.bse.to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(fit.resid, native.resid, atol=1e-10)
    np.testing.assert_allclose(fit.fitted, res.fittedvalues, atol=1e-10)
    assert fit.stats["r2"] == pytest.approx(res.rsquared) and fit.stats["df_resid"] == res.df_resid
    summary = fit.summary().to_frame()
    np.testing.assert_allclose(summary["p>|t|"], res.pvalues, rtol=1e-8)  # the same t statistics and p values
    prov = fit.provenance
    assert prov.method["model"] == "OLS" and prov.method["cov_type"] == "nonrobust"
    assert prov.method["backend"].startswith("statsmodels") and prov.spec["command"] == "sm.OLS(y, X).fit()"
    assert prov.data_fingerprint and prov.sample == ("2000Q2", "2024Q4")
    assert list(fit.to_frame("coefficients").index) == ["const", "y", "c[-1]"]
    robust = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": True})
    hac = FittedEquation.from_statsmodels(robust, data)
    assert hac.provenance.resolved == {"hac_lags": 4} and "HAC" in hac.summary().notes[0]
    own = Equation("c ~ y + c[-1]").fit(data, method=OLS(cov="hac", hac_lags=4)).params.se
    np.testing.assert_allclose(hac.params.se, own, rtol=1e-8)


def test_the_bridge_runs_headless():
    pytest.importorskip("statsmodels")
    code = textwrap.dedent("""
        import sys
        import numpy as np
        import statsmodels.api as sm
        from pymetrix import Dataset
        from pymetrix.estimation import FittedEquation
        data = Dataset.example("sample_macro").apply_np("lc", np.log, "c", source_expr="np.log(c)")
        y, X = data.design("lc", ["log(y)", "lc[-1]"])
        fit = FittedEquation.from_statsmodels(sm.OLS(y, X).fit(), data)
        print(fit.summary().to_markdown().splitlines()[0])
        print("LOADED", sorted(m for m in sys.modules if m.split(".")[0] in ("PySide6", "shiboken6", "qtpy", "matplotlib")))
    """)
    env = {k: v for k, v in os.environ.items() if k not in ("QT_QPA_PLATFORM", "DISPLAY")}
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    assert "LOADED []" in run.stdout
