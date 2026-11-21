"""Three ways data could change meaning silently, now caught: a new active sample leaves results computed on the old
one up to date; adding a series under an existing name overwrites it (keeping the old series' source); a series of
another frequency becomes a column of missing values."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, DataError
from pymetrix.commands import Session
from pymetrix.commands.builtin import AddToDataset, DeriveSeries, FitEquation, LoadExample
from pymetrix.commands.data_edit import GenerateSeries
from pymetrix.commands.registry import upgrade
from pymetrix.commands.samples import SetSample
from pymetrix.commands.selection import DeriveEach
from pymetrix.metadata import Vintage


@pytest.fixture
def session():
    s = Session()
    s.run(LoadExample())
    return s


def test_a_new_active_sample_makes_results_out_of_date(session):
    session.run(FitEquation("c ~ y", name="fit"))
    before = session["fit"].provenance
    assert session.stale() == {}
    session.run(SetSample("data", start="2010Q1", end="2024Q4"))
    assert session.stale() == {"fit": "the active sample of data has changed since fit was computed"}
    assert session["fit"].provenance is before and before.sample == ("2000Q1", "2024Q4")  # the old result is as it was
    session.recompute()
    assert session["fit"].provenance.sample == ("2010Q1", "2024Q4") and session.stale() == {}
    session.run(GenerateSeries("ratio", "c / y"))  # adding a series is not a change of sample
    assert session.stale() == {}


@pytest.mark.parametrize("make", [
    lambda: GenerateSeries("y", "y * 2"),
    lambda: DeriveSeries("y", "c + i"),
    lambda: DeriveEach("{s}", "{s} * 2", series="c, y", keep_updated=False),
])
def test_adding_a_series_under_an_existing_name_needs_replace(session, make):
    step = make()
    with pytest.raises(Exception, match="already has .*tick Replace"):
        session.run(step)
    assert np.allclose(session["data"]["y"], Dataset.example("sample_macro")["y"])  # nothing changed
    session.run(replace(step, replace=True))
    assert not np.allclose(session["data"]["y"], Dataset.example("sample_macro")["y"])


def test_adding_a_fitted_series_under_an_existing_name_needs_replace(session):
    session.run(FitEquation("c ~ y", name="fit"))
    session.run(AddToDataset("fit", "resid", series_name="res"))
    with pytest.raises(Exception, match="already has a series res"):
        session.run(AddToDataset("fit", "resid", series_name="res"))
    session.run(AddToDataset("fit", "resid", series_name="res", replace=True))


def test_steps_saved_before_the_option_keep_replacing():
    for cid in ("pymetrix:add_to_dataset", "pymetrix:generate_series", "pymetrix:derive_series", "pymetrix:derive_each"):
        assert upgrade(cid, 1, {})["replace"] is True


def test_replaced_values_keep_the_label_but_not_the_source():
    data = Dataset.example("sample_macro")
    sourced = replace(data.variables["y"], source="FRED GDPC1", vintage=Vintage(pd.Timestamp("2024-01-25").date()))
    data = Dataset(data.frame, data.axes, {**data.variables, "y": sourced}, data.flags, data.meta)
    out = data.with_series(y=data["y"] * 2)
    v = out.variables["y"]
    assert v.label == data.variables["y"].label and v.source == "" and v.vintage is None
    kept = data.with_series(z=data["c"])  # a new name: nothing to drop
    assert kept.variables["y"].source == "FRED GDPC1"


def test_a_series_of_another_frequency_is_refused():
    data = Dataset.example("sample_macro")  # quarterly
    monthly = pd.Series(np.arange(300.0), index=pd.period_range("2000-01", periods=300, freq="M"))
    with pytest.raises(DataError, match="is M, but this dataset is Q"):
        data.with_series(m=monthly)
    elsewhere = pd.Series(1.0, index=pd.period_range("1900Q1", periods=8, freq="Q"))
    with pytest.raises(DataError, match="none of the periods"):
        data.with_series(old=elsewhere)
    same = pd.Series(np.arange(len(data.frame), dtype=float), index=data.frame.index)
    assert data.with_series(n=same)["n"].notna().all()
