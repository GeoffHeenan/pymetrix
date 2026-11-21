"""Samples: named, combinable selections of periods, entities and rows, and a dataset's active sample."""

import io

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, Sample
from pymetrix.io.parquet import read_parquet, write_parquet


@pytest.fixture
def series():
    idx = pd.period_range("2000Q1", periods=40, freq="Q", name="time")
    rng = np.random.default_rng(0)
    x = rng.normal(size=40).cumsum()
    return Dataset.from_frame(pd.DataFrame({"y": 0.5 * x + rng.normal(size=40), "x": x}, index=idx))


@pytest.fixture
def panel():
    rng = np.random.default_rng(1)
    rows = [{"country": c, "year": y, "x": rng.normal(), "pop": p}
            for c, p in (("DEU", 80.0), ("FRA", 65.0), ("MLT", 0.5)) for y in range(1990, 2010)]
    frame = pd.DataFrame(rows)
    frame["y"] = 1.0 + 2.0 * frame["x"] + 0.1 * rng.normal(size=len(frame))
    return Dataset.from_frame(frame, entity="country", time="year", freq="Y")


def test_samples_are_values_that_combine_print_and_store(series):
    s = Sample(("2003Q1", "2007Q4"))
    assert s == Sample(start="2003Q1", end="2007Q4") and hash(s) == hash(Sample(start="2003Q1", end="2007Q4"))
    assert repr(s) == "Sample(('2003Q1', '2007Q4'))" and str(s) == "2003Q1 to 2007Q4"
    assert str(Sample(end="2005Q4") | Sample(start="2008Q1")) == "(to 2005Q4) or (from 2008Q1)"
    both = Sample(end="2005Q4") & ~Sample(where="x > 0")
    assert repr(both) == "(Sample(end='2005Q4') & ~Sample(where='x > 0'))"
    assert Sample.from_dict(both.to_dict()) == both
    mask = both.mask(series)
    expected = (series.frame.index <= pd.Period("2005Q4")) & ~(series["x"] > 0).to_numpy()
    assert np.array_equal(mask, expected)
    assert s.describe(series) == "20 of 40 rows, 2003Q1 to 2007Q4"
    with pytest.raises(AttributeError):
        s.start = "2001Q1"
    with pytest.raises(ValueError, match="not both"):
        Sample(("2001Q1", "2002Q1"), start="2001Q1")
    with pytest.raises(ValueError, match="a pair"):
        Sample("2001Q1")
    assert Sample().everything and not s.everything


def test_the_active_sample_is_what_estimation_uses_unless_told_otherwise(series):
    data = series.with_sample(Sample(("2003Q1", "2007Q4")))
    assert data.sample == Sample(("2003Q1", "2007Q4")) and series.sample is None
    fit = Equation("y ~ x").fit(data)
    explicit = Equation("y ~ x").fit(series, span=("2003Q1", "2007Q4"))
    assert fit.provenance.sample == ("2003Q1", "2007Q4") and fit.provenance.nobs == 20
    assert np.allclose(fit.params.values, explicit.params.values)
    assert Equation("y ~ x").fit(data, span=("2000Q1", "2001Q4")).provenance.nobs == 8  # an explicit span replaces it
    assert Equation("y ~ x").fit(data, span=Sample(end="2001Q4")).provenance.nobs == 8
    lagged = Equation("y ~ x[-1]").fit(data)
    assert lagged.provenance.nobs == 20  # the first period of the sample uses the lag from before it
    assert data.with_sample(None).sample is None and data.with_sample(Sample()).sample is None
    assert data.with_sample(end="2001Q4").sample == Sample(end="2001Q4")
    assert data.with_sample(("2001Q1", "2001Q4")).sample_mask().sum() == 4
    with pytest.raises(ValueError, match="selects no rows"):
        series.with_sample(start="2030Q1")
    derived = data.derive(dx="d(x)")
    assert derived.rename(x="z").sample == data.sample  # transformations keep the active sample
    assert derived["dx"].notna().sum() == 39  # and are computed over all the data, not just the sample


def test_tables_graphs_and_the_data_window_follow_the_sample(series):
    from pymetrix.views import dataset_status

    data = series.with_sample(end="2001Q4")
    table = data.describe()
    assert table.rows[0][0] == 8 and "Sample: to 2001Q4 (8 of 40 rows" in table.notes[0]
    assert data.corr().notes and not series.corr().notes
    assert len(data.plot().panels[0].layers[0].y) == 8
    assert "sample to 2001Q4: 8 periods shown" in dataset_status(data)
    assert len(data.in_sample()) == 8 and len(data.frame) == 40  # all the data are kept


def test_panels_entities_windows_and_conditions(panel):
    core = Sample((1995, 2004), country=["DEU", "FRA"])
    data = panel.with_sample(core)
    assert core.describe(panel) == "20 of 60 rows, 2 country, 1995 to 2004"
    fit = Equation("y ~ x").fit(data)
    assert fit.provenance.nobs == 20
    large = Sample(where="pop > 1")
    assert (core & large).mask(panel).sum() == 20 and (core | large).mask(panel).sum() == 40
    windows = Sample(windows={"DEU": (2000, None), "*": (2005, None)})
    mask = windows.mask(panel)
    counts = panel.frame[mask].groupby(level="country").size().to_dict()
    assert counts == {"DEU": 10, "FRA": 5, "MLT": 5}
    lagged = Equation("y ~ x[-1]").fit(panel.with_sample(country=["DEU", "FRA"]))
    assert lagged.provenance.nobs == 38  # lags within each country, never across: one lost per country
    with pytest.raises(ValueError, match="windows= needs one entity axis"):
        Sample(windows={"*": ("2000Q1", None)}).mask(Dataset.from_frame(pd.DataFrame(
            {"x": [1.0, 2.0]}, index=pd.period_range("2000Q1", periods=2, freq="Q", name="time"))))


def test_the_sample_is_saved_with_the_data(series, tmp_path):
    data = series.with_sample(Sample(end="2005Q4") & Sample(where="x > -100"))
    buffer = io.BytesIO()
    write_parquet(data, buffer)
    assert read_parquet(io.BytesIO(buffer.getvalue())).sample == data.sample

    from pymetrix.commands import NewSample, Session, SetSample, load_project, save_project

    s = Session()
    s.objects["data"] = series
    s.run(NewSample("precrisis", start="2001Q1", end="2005Q4"))
    s.run(SetSample(sample="precrisis"))
    s.run(NewSample("rich", entities="", where="x > 0"))
    assert "precrisis = Sample(('2001Q1', '2005Q4'))" in s.script() and "data = data.with_sample(precrisis)" in s.script()
    assert s["data"].sample == Sample(("2001Q1", "2005Q4"))
    assert s.sample_bounds({"data": "data"}) == ("2001Q1", "2005Q4")
    s.run(SetSample(start="2002Q1"))
    assert "data = data.with_sample(Sample(start='2002Q1'))" in s.script()
    s.run(SetSample())
    assert s["data"].sample is None and "data = data.with_sample(None)" in s.script()
    path = save_project(s, tmp_path / "samples.pymx")
    reopened = load_project(path)
    assert reopened["precrisis"] == Sample(("2001Q1", "2005Q4")) and reopened["data"].sample is None
    assert SetSample(sample="precrisis", start="2001Q1").validate(s) == ["choose a named sample or give periods, not both"]
    assert NewSample("bad name").validate(s) == ["'bad name' is not a valid name"]
    assert NewSample("none", start="2030Q1").validate(s) == ["the sample selects no rows of data"]
