"""Results record what was used, not only what was asked for: choices made automatically (HAC lags, lags chosen by a
criterion, a test's lag), how the rows were chosen, the indicators' vintages, each data load, and the filter engine."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import MIDAS, OLS, VAR, Dataset, Equation, Sample, demo
from pymetrix.metadata import Vintage


@pytest.fixture
def data():
    return Dataset.example("sample_macro")


def test_automatic_choices_are_recorded(data):
    fit = Equation("c ~ y").fit(data, method=OLS(cov="hac"))
    assert fit.provenance.method["hac_lags"] is None and fit.provenance.resolved == {"hac_lags": int(fit.stats["hac_lags"])}
    assert Equation("c ~ y").fit(data).provenance.resolved is None  # nothing chosen automatically
    var = VAR(["c", "y"], lags="aic").fit(data)
    assert var.provenance.resolved == {"lags": var.lags, "chosen by": "aic"}


def test_a_tests_own_choices_are_recorded(data):
    pytest.importorskip("statsmodels")  # the ADF test is statsmodels'
    adf = data.test("adf", "y")
    assert adf.provenance.method["settings"]["lags"] == "aic" and "lags used" in adf.provenance.resolved


def test_how_the_rows_were_chosen(data):
    assert Equation("c ~ y").fit(data).provenance.sample_source == "all observations"
    explicit = Equation("c ~ y").fit(data, span=("2010Q1", None)).provenance
    assert explicit.sample_source == "explicit" and explicit.sample_definition == {"span": ["2010Q1", None]}
    rule = Sample(("2005Q1", "2020Q4"), where="y > 100")
    active = Equation("c ~ y").fit(data.with_sample(rule)).provenance
    assert active.sample_source == "active sample of the dataset" and active.sample_definition["where"] == "y > 100"
    assert active.sample == ("2005Q1", "2020Q4")  # the end dates alone would lose the condition


def test_midas_uses_the_active_sample_and_records_the_indicators_vintages():
    rng = np.random.default_rng(0)
    months = pd.period_range("2000-01", periods=240, freq="M", name="time")
    quarters = pd.period_range("2000Q1", periods=80, freq="Q", name="time")
    ip = pd.Series(rng.normal(size=240), index=months)
    gdp = pd.Series(ip.groupby(ip.index.asfreq("Q")).mean().to_numpy() + 0.1 * rng.normal(size=80), index=quarters)
    low = Dataset.from_frame(pd.DataFrame({"gdp": gdp}))
    high = Dataset.from_frame(pd.DataFrame({"ip": ip}))
    from dataclasses import replace

    high = Dataset(high.frame, high.axes, {"ip": replace(high.variables["ip"], vintage=Vintage(pd.Timestamp("2020-02-14").date()))},
                   high.flags, high.meta)
    spec = MIDAS("gdp ~ ip", lags=6)
    whole = spec.fit(low, high)
    narrowed = spec.fit(low.with_sample(Sample(("2010Q1", None))), high)
    assert narrowed.provenance.nobs < whole.provenance.nobs and narrowed.provenance.sample[0] == "2010Q1"
    assert narrowed.provenance.sample_source == "active sample of the dataset"
    assert narrowed.provenance.data_vintages["ip"].released == pd.Timestamp("2020-02-14").date()


def test_each_load_is_recorded(tmp_path):
    path = tmp_path / "rates.csv"
    pd.DataFrame({"time": ["2020Q1", "2020Q2", "2020Q3"], "r": [1.0, 2.0, 3.0]}).to_csv(path, index=False)
    loaded = Dataset.read(path, time="time", freq="Q")
    (entry,) = loaded.meta.extra["loads"]
    assert entry["source"] == str(path) and entry["options"] == {"time": "time", "freq": "Q"} and entry["retrieved"]
    loaded.write(tmp_path / "rates.parquet")
    assert Dataset.read(tmp_path / "rates.parquet").meta.extra["loads"][0]["source"] == str(path)  # kept, then added to
    assert len(Dataset.read(tmp_path / "rates.parquet").meta.extra["loads"]) == 2


def test_scripts_name_the_filter_engine_when_one_was_chosen(monkeypatch):
    from pymetrix import statespace

    session = demo("consumption")
    assert "ENGINE" not in session.script()
    monkeypatch.setattr(statespace, "ENGINE", "native")
    assert "pymetrix.statespace.ENGINE = 'native'" in session.script()
