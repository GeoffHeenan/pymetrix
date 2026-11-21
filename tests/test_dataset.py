from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Measure, Variable, Vintage


def test_kinds(ts, panel):
    assert ts.kind == "time_series"
    assert panel.kind == "panel"
    cross = Dataset.from_frame(pd.DataFrame({"hh": [1, 2, 3], "inc": [1.0, 2.0, 3.0]}), entity="hh")
    assert cross.kind == "cross_section"


def test_lags_stay_within_entity(panel):
    lagged = panel.lagged("x", -1)
    first_rows = lagged.groupby(level="country").head(1)
    assert first_rows.isna().all()
    c01 = panel.select(country="C01")
    assert np.allclose(lagged.xs("C01", level="country").iloc[1:], c01["x"].iloc[:-1])


def test_lag_on_cross_section_fails():
    cross = Dataset.from_frame(pd.DataFrame({"hh": [1, 2], "inc": [1.0, 2.0]}), entity="hh")
    with pytest.raises(ValueError):
        cross.eval("inc[-1]")


def test_eval_and_span(ts):
    s = ts.eval("dlog(yd)")
    assert np.isclose(s.iloc[1], np.log(ts["yd"].iloc[1] / ts["yd"].iloc[0]))
    sub = ts.select(span=("2010Q1", "2010Q4"))
    assert len(sub) == 4


def test_to_freq_uses_measures(ts):
    variables = {"y": Variable("y", measure=Measure.FLOW), "c": Variable("c", measure=Measure.STOCK)}
    small = Dataset(ts.frame[["y", "c"]], ts.axes, variables)
    annual = small.to_freq("Y")
    assert np.isclose(annual["y"].iloc[0], ts["y"].iloc[:4].sum())
    assert np.isclose(annual["c"].iloc[0], ts["c"].iloc[3])
    partial = small.select(span=("2000Q2", None)).to_freq("Y")
    assert np.isnan(partial["y"].iloc[0])
    with pytest.raises(ValueError, match="no measure"):
        ts.to_freq("Y")


def test_aggregate_and_pivot(panel):
    variables = {n: Variable(n, measure=Measure.FLOW) for n in panel.names if n != "region"}
    p = Dataset(panel.frame, panel.axes, variables)
    totals = p.aggregate(by="year")
    assert np.isclose(totals["x"].iloc[0], panel.frame.xs(pd.Period("2000", "Y"), level="year")["x"].sum())
    table = panel.pivot(rows="region", columns="year", values="x", agg="mean")
    assert table.index == ["A", "B"]


def test_parquet_round_trip_keeps_metadata(ts, tmp_path):
    v = Variable("c", units="WST", scale=6, measure=Measure.FLOW, vintage=Vintage(date(2026, 9, 30)), symbol="C_t")
    ds = Dataset(ts.frame, ts.axes, {"c": v})
    path = tmp_path / "data.parquet"
    ds.write(str(path))
    back = Dataset.read(str(path))
    assert back.variables["c"] == v
    assert back.frame.index.equals(ts.frame.index)
    assert np.allclose(back["c"], ts["c"])


def test_csv_round_trip(panel, tmp_path):
    path = tmp_path / "panel.csv"
    panel.write(str(path))
    back = Dataset.read(str(path), entity="country", time="year", freq="Y")
    assert back.kind == "panel"
    assert np.allclose(back["y"], panel["y"])


def test_lags_by_position_equal_lags_by_period():
    """A run of consecutive periods is shifted by position; with gaps, or out of order, by period: the
    same answer either way."""
    import numpy as np
    import pandas as pd

    from pymetrix import Dataset

    rng = np.random.default_rng(3)
    periods = pd.period_range("2000Q1", periods=12, freq="Q", name="time")
    cases = {"consecutive": periods, "a gap": periods.delete(5), "out of order": periods[[1, 0, *range(2, 12)]]}
    for label, index in cases.items():
        values = rng.normal(size=len(index))
        values[3] = np.nan
        data = Dataset.from_frame(pd.DataFrame({"x": values}, index=index))
        for lag in (-1, -3, 2, -11, -12, 13):
            got = data.lagged("x", lag)
            by_period = pd.Series(data["x"].to_numpy(), index=data["x"].index - lag).reindex(data["x"].index)
            assert got.name == "x" and got.index.equals(data["x"].index), (label, lag)
            np.testing.assert_array_equal(got.to_numpy(), by_period.to_numpy(), err_msg=f"{label}, lag {lag}")
