"""A time axis from a first period and a frequency, for data with no dates (R's ``ts(x, start = c(1990, 1),
frequency = 4)``): ``Dataset.from_frame(frame, start=..., freq=...)``, ``Dataset.read(..., start=..., freq=...)`` and
File → Load data…'s Dates start at. Checked against pandas' ``period_range`` written out by hand, for years, quarters,
months and weeks, and through a Parquet round trip that keeps the axis and the series' metadata."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset
from pymetrix.errors import DataError


def _frame(n=8):
    return pd.DataFrame({"y": np.arange(n, dtype=float), "x": np.linspace(0, 1, n)})


@pytest.mark.parametrize("start, freq, first, last, axis_freq", [
    ("1990", "Y", "1990", "1997", "Y-DEC"),
    ("1990Q1", "Q", "1990Q1", "1991Q4", "Q-DEC"),
    ("1990Q3", "Q", "1990Q3", "1992Q2", "Q-DEC"),
    ("1990", "Q", "1990Q1", "1991Q4", "Q-DEC"),  # R's start = 1990 with frequency 4
    ("1990-01", "M", "1990-01", "1990-08", "M"),
    ("1990-11", "M", "1990-11", "1991-06", "M"),
    ("1990-01-01", "W", "1990-01-01/1990-01-07", "1990-02-19/1990-02-25", "W-SUN"),
    ("1990", "Y-JUN", "1990", "1997", "Y-JUN"),
])
def test_periods_are_pandas_period_range(start, freq, first, last, axis_freq):
    data = Dataset.from_frame(_frame(), start=start, freq=freq)
    expected = pd.period_range(start=pd.Period(first, freq=axis_freq), periods=8, freq=axis_freq)
    assert list(data.frame.index) == list(expected)
    assert str(data.frame.index[0]) == first and str(data.frame.index[-1]) == last
    assert data.kind == "time_series" and data.time_axis.name == "time" and data.time_axis.freq == axis_freq
    assert list(data["y"]) == list(range(8))  # one period per row, in order


def test_integer_index_is_replaced_and_others_refused():
    years = _frame(3).set_index(pd.Index([7, 8, 9]))
    assert str(Dataset.from_frame(years, start="2001", freq="Y").frame.index[0]) == "2001"
    with pytest.raises(DataError, match="time="):
        Dataset.from_frame(_frame(3).set_index(pd.Index(["a", "b", "c"])), start="2001", freq="Y")


@pytest.mark.parametrize("start, freq, message", [
    ("199x", "Q", "not the start of a period"),
    ("1990-03", "Q", "1990Q1 starts 1990-01-01"),  # inside a quarter: refused, not rounded down
    ("1990-01-03", "W", "not the start"),  # a Wednesday: weeks end on Sunday
    ("1990Q1", None, "needs freq="),
    ("1990Q1", "fortnight", "not a frequency"),
])
def test_starts_that_do_not_parse_are_refused(start, freq, message):
    with pytest.raises(DataError, match=message):
        Dataset.from_frame(_frame(), start=start, freq=freq)


def test_start_with_a_time_column_is_refused():
    frame = _frame(3).assign(date=["2000", "2001", "2002"])
    with pytest.raises(DataError, match="leave it out"):
        Dataset.from_frame(frame, time="date", freq="Y", start="2000")


def test_numpy_array_through_a_frame():
    array = np.column_stack([np.arange(4.0), np.ones(4)])
    data = Dataset.from_frame(pd.DataFrame(array, columns=["y", "x"]), start="2020-01", freq="M", source="user:numpy")
    assert str(data.frame.index[-1]) == "2020-04" and data.variables["y"].source == "user:numpy"


def test_parquet_round_trip_keeps_axis_and_metadata(tmp_path):
    pytest.importorskip("pyarrow")
    data = Dataset.from_frame(_frame(), start="1990Q1", freq="Q", source="user:test").with_info("y", label="Output")
    path = tmp_path / "dated.parquet"
    data.write(str(path))
    back = Dataset.read(str(path))
    pd.testing.assert_frame_equal(back.frame, data.frame)
    assert back.axes == data.axes
    assert back.variables["y"].label == "Output" and back.variables["x"].source == "user:test"


def test_read_csv_without_dates_and_load_data_dialog(tmp_path):
    from pymetrix.commands import LoadData, Session

    path = tmp_path / "nodates.csv"
    _frame(6).to_csv(path, index=False)
    data = Dataset.read(str(path), start="1990Q1", freq="Q")
    assert str(data.frame.index[0]) == "1990Q1" and str(data.frame.index[-1]) == "1991Q2"

    step = LoadData(str(path), freq="Q", first_period="1990Q1", name="d")
    s = Session()
    assert step.validate(s) == []
    assert "start='1990Q1'" in step.to_code() and "freq='Q'" in step.to_code()
    s.run(step)
    pd.testing.assert_frame_equal(s["d"].frame, data.frame)
    assert LoadData(str(path), first_period="1990Q1").validate(s) == ["choose the frequency of the periods that Dates start at begins"]
    assert "Time" in LoadData(str(path), time="y", freq="Q", first_period="1990Q1").validate(s)[0]
    assert "start of a period" in LoadData(str(path), freq="Q", first_period="1990-02").validate(s)[0]
    assert LoadData(str(path)).validate(s) == []  # left empty: the file's own dates, as before
