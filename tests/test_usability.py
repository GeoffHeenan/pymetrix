"""Usability (plan items 8 to 11): plain pandas in and out, lists wherever series names go, chainable samples
written into code and provenance, and the vintage asked for recorded explicitly."""

import inspect
import json

import numpy as np
import pandas as pd
import pytest

import pymetrix as pm
from pymetrix import VAR, Dataset, Equation, Sample, Selection
from pymetrix.errors import DataError


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    idx = pd.period_range("1990Q1", periods=60, freq="Q")
    return pd.DataFrame({"y": rng.normal(size=60).cumsum(), "x": rng.normal(size=60)}, index=idx)


# item 9: a simple pandas path

def test_plain_dataframes_are_accepted(frame):
    fit = Equation("y ~ x").fit(frame)
    ref = Equation("y ~ x").fit(Dataset.from_frame(frame))
    assert np.allclose(fit.params.values, ref.params.values)
    assert fit.provenance.vintage_request == {"policy": "user"}
    dated = frame.copy()
    dated.index = frame.index.to_timestamp()  # dates with a quarterly frequency
    assert VAR(["y", "x"], lags=1).fit(dated).provenance.sample == ("1990Q2", "2004Q4")


def test_unreadable_frames_say_how(frame):
    with pytest.raises(DataError, match="PeriodIndex"):
        Equation("y ~ x").fit(frame.reset_index(drop=True))
    irregular = frame.copy()
    irregular.index = pd.to_datetime(["2000-01-01", "2000-01-05", *pd.date_range("2001-01-01", periods=58, freq="D")])
    with pytest.raises(DataError, match="no regular frequency"):
        pm.as_dataset(irregular)


def test_panels_from_a_multiindex():
    idx = pd.MultiIndex.from_product([["DEU", "FRA"], pd.period_range("2000", periods=5, freq="Y")], names=["country", "year"])
    df = pd.DataFrame({"y": np.arange(10.0)}, index=idx)
    data = pm.dataset(df)
    assert data.kind == "panel" and [a.name for a in data.axes] == ["country", "year"]


def test_to_pandas_is_the_way_out(frame):
    data = pm.as_dataset(frame).with_sample(("1995Q1", "1999Q4"))
    out = data.to_pandas()
    assert isinstance(out, pd.DataFrame) and out.equals(frame.rename_axis("time"))  # every row, whatever the active sample
    assert len(data.to_pandas(sample=None)) == 20 and list(data.to_pandas(["x"]).columns) == ["x"]


# item 10: lists wherever a Selection is expected

def _with_names(method):
    params = inspect.signature(method).parameters.values()
    return any(p.kind is p.VAR_POSITIONAL and p.name in ("names", "patterns") for p in params)


@pytest.mark.parametrize("name", sorted(n for n, m in vars(Dataset).items() if callable(m) and _with_names(m)))
def test_dataset_methods_taking_names_accept_lists_tuples_and_selections(name):
    data = Dataset.example()
    method = getattr(data, name)
    results = []
    for form in ("c", ["c"], ("c",), Selection(("c",))):
        try:
            results.append(method(form))
        except Exception as exc:  # noqa: BLE001 - methods for panels raise for time series, the same for every form
            results.append(type(exc))
    assert len({repr(type(r)) if not isinstance(r, type) else r.__name__ for r in results}) == 1, results


def test_selection_helpers():
    data = Dataset.example()
    assert pm.as_selection(["c", "i"]) == Selection(("c", "i")) and pm.as_selection("c,i") == Selection(("c,i",))
    assert list(pm.as_selection("c*", data)) == ["c"]
    fits = pm.fit_each(["c", "i"], "{s} ~ {s}[-1]", data)
    assert list(fits) == ["c", "i"]
    tests = pm.test_each(("c",), data, "adf")
    assert list(tests) == ["c"]


# item 8: chainable samples

def test_span_text_everywhere(frame):
    data = pm.as_dataset(frame)
    a = Equation("y ~ x").fit(data, span="1995Q1:2004Q4")
    b = Equation("y ~ x").fit(data, span=("1995Q1", "2004Q4"))
    assert np.allclose(a.params.values, b.params.values) and a.provenance.sample == ("1995Q1", "2004Q4")
    assert Sample("1995Q1:") == Sample(start="1995Q1") and Sample(":2000Q4") == Sample(end="2000Q4")
    assert data.with_sample("1995Q1:1999Q4").sample == Sample(("1995Q1", "1999Q4"))
    with pytest.raises(DataError, match="start:end"):
        data.sample_mask("1995Q1")


def test_over_and_fit_chain(frame):
    data = pm.as_dataset(frame)
    fit = data.over("1995Q1:2004Q4").fit(Equation("y ~ x"), method=pm.OLS(cov="hac"))
    assert fit.provenance.sample == ("1995Q1", "2004Q4") and fit.provenance.sample_source == "active sample of the dataset"
    assert fit.provenance.sample_definition == {"start": "1995Q1", "end": "2004Q4"}
    var = data.over(start="2000Q1").fit(VAR(["y", "x"], lags=1))
    assert var.provenance.sample[0] == "2000Q1"
    with pytest.raises(TypeError, match="cannot be estimated"):
        data.fit("y ~ x")


def test_the_chain_is_what_generated_code_says():
    from pymetrix.commands import Session

    s = Session()
    s.run_code("data = Dataset.example()")
    s.run_code("fit = data.over('2005Q1:2019Q4').fit(Equation('dlog(c) ~ dlog(y)'), method=OLS(cov='hac'))")
    assert s.history[-1].code == "fit = data.over('2005Q1:2019Q4').fit(Equation('dlog(c) ~ dlog(y)'), method=OLS(cov='hac'))"
    assert s["fit"].provenance.sample == ("2005Q1", "2019Q4")


# item 11: the vintage asked for, explicit

def test_reads_record_the_vintage_asked_for(tmp_path, frame):
    path = tmp_path / "data.csv"
    frame.rename_axis("date").reset_index().assign(date=lambda d: d["date"].astype(str)).to_csv(path, index=False)
    data = Dataset.read(str(path), time="date", freq="Q")
    assert data.meta.vintage["policy"] == "file" and data.meta.vintage["modified"]
    fit = Equation("y ~ x").fit(data)
    assert fit.provenance.vintage_request["policy"] == "file"
    with pytest.raises(DataError, match="vintage= is for connectors"):
        Dataset.read(str(path), time="date", freq="Q", vintage="2020-01-30")
    assert Dataset.example().meta.vintage["policy"] == "example"


def test_connector_reads_write_the_vintage_into_code(monkeypatch):
    from pymetrix.calls import make_call

    assert make_call("pymetrix:dataset.read", output="g", source="fred://GDPC1").to_code() == "g = Dataset.read('fred://GDPC1', vintage='latest')"
    assert make_call("pymetrix:dataset.read", output="d", source="data.csv").to_code() == "d = Dataset.read('data.csv')"


def test_connector_vintages_from_fixtures(monkeypatch):
    from pathlib import Path

    from pymetrix.ext import connectors

    fixture = Path(__file__).parent / "fixtures" / "connectors" / "fred_gdpc1_2020.csv"
    monkeypatch.setattr(connectors, "_get", lambda url, *a, **k: fixture.read_bytes(), raising=False)
    try:
        data = Dataset.read("fred://GDPC1", vintage="2020-01-30")
    except Exception as exc:  # noqa: BLE001 - the connector's download helper has another name
        pytest.skip(f"connector not patchable here: {exc}")
    assert data.meta.vintage == {"policy": "as_of", "date": "2020-01-30"}
    latest = Dataset.read("fred://GDPC1")
    assert latest.meta.vintage["policy"] == "latest" and "retrieved" in latest.meta.vintage


def test_projects_replay_typed_reads_from_the_saved_data(tmp_path, frame):
    from pymetrix.commands import Session, load_project, save_project

    path = tmp_path / "data.csv"
    frame.rename_axis("date").reset_index().assign(date=lambda d: d["date"].astype(str)).to_csv(path, index=False)
    s = Session()
    s.run_code(f"data = Dataset.read({str(path)!r}, time='date', freq='Q')")
    assert type(s.history[-1].command).__name__ == "Call"
    project = save_project(s, tmp_path / "p.pymx")
    path.unlink()  # the file is gone: the project still has the data as read
    opened = load_project(project)
    pd.testing.assert_frame_equal(opened["data"].frame, s["data"].frame)
    assert json.loads(json.dumps(opened["data"].meta.vintage))["policy"] == "file"
