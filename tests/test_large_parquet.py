"""Reading part of a large Parquet file: columns, row conditions and spans pushed down to pyarrow.

Every selection is checked against pandas reading the whole file and selecting (exact equality), and the pruning
against pyarrow's own metadata: the row groups pyarrow's statistics keep are those read."""

import time
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import pytest

from pymetrix import Dataset
from pymetrix.errors import DataError, SpecificationError
from pymetrix.io import scan
from pymetrix.io.pushdown import condition_expression
from pymetrix.io.selection import fingerprint


def _panel(n=60_000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"id": np.arange(n) // 20, "year": 1990 + np.arange(n) % 20, "x": rng.normal(size=n),
                       "country": rng.choice(["DEU", "FRA", "ITA"], n), "y": rng.normal(size=n),
                       "flag": rng.integers(0, 2, n).astype(bool),
                       "day": pd.Timestamp("2000-01-01") + pd.to_timedelta(np.arange(n) % 3000, "D")})
    df.loc[::7, "y"] = np.nan  # missing values: pandas keeps them for != and not in, pyarrow's filter does not
    return df


@pytest.fixture(scope="module")
def panel_file(tmp_path_factory):
    folder = tmp_path_factory.mktemp("big")
    df = _panel()
    path = folder / "panel.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, row_group_size=3_000)  # 20 row groups, sorted by id
    return path, df


def _whole(path, where=None, span=None, columns=None, **kw):
    """The reference: pandas reads everything, then the selection is made in memory."""
    data = Dataset.from_frame(pd.read_parquet(path), **kw)
    if where is not None or span is not None:
        data = data.select(where=where, span=span)
    return data.frame if columns is None else data.frame[columns]


CONDITIONS = [
    "id < 500",
    "id >= 2000 and id < 2100 and x > 0",
    "country in ['DEU', 'ITA'] and y != 0",
    "country not in ['FRA'] or y > 1",
    "(x > 1) | (y < -1)",
    "not (x > 0)",
    "y == y",  # two columns: left to the exact evaluation
    "log(abs(x)) > 0 and id < 300",
    "flag == True and 100 <= id < 400",
    "day >= '2005-06-01' and id > 1000",
    "x > 2.5 and id <= 2999.0",
    "year >= '2001' and year < '2004' and id < 1000",  # the time column, as periods
    "year != '2001' and day != '2001-01-01' and id < 300",  # != on periods and dates: pandas takes it as always true
    "day == '2001-01-01' or day in ['2002-01-01']",
]


@pytest.mark.parametrize("where", CONDITIONS)
def test_a_condition_reads_the_rows_pandas_selects(panel_file, where):
    path, _ = panel_file
    kw = dict(entity="id", time="year", freq="Y")
    got = Dataset.read(str(path), where=where, **kw)
    pd.testing.assert_frame_equal(got.frame, _whole(path, where, **kw))


def test_the_condition_matches_a_plain_pandas_mask(panel_file):
    """The same rows as a mask written in pandas by hand, not through pymetrix's conditions."""
    path, df = panel_file
    got = Dataset.read(str(path), where="country in ['DEU'] and y != 0 and id < 1500", entity="id", time="year", freq="Y",
                       columns=["y"])
    mask = df.country.isin(["DEU"]) & (df.y != 0) & (df.id < 1500)  # NaN != 0 is True in pandas
    expected = df[mask].set_index(["id", "year"])["y"].sort_index()
    assert got.frame["y"].to_numpy(dtype=float).tobytes() == expected.to_numpy(dtype=float).tobytes()
    assert list(got.frame.index.get_level_values("id")) == list(expected.index.get_level_values("id"))


def test_columns_and_span(panel_file):
    path, _ = panel_file
    kw = dict(entity="id", time="year", freq="Y")
    got = Dataset.read(str(path), columns=["x"], span=("1995", "1999"), **kw)
    assert list(got.frame.columns) == ["x"]
    pd.testing.assert_frame_equal(got.frame, _whole(path, span=("1995", "1999"), columns=["x"], **kw))
    # a condition on a column not kept: read for the condition, then dropped
    got = Dataset.read(str(path), columns="x", where="y > 0", span="1995:", **kw)
    assert list(got.frame.columns) == ["x"]
    pd.testing.assert_frame_equal(got.frame, _whole(path, "y > 0", ("1995", None), ["x"], **kw))


def test_a_sorted_column_skips_row_groups_as_pyarrow_says(panel_file):
    path, _ = panel_file
    got = Dataset.read(str(path), where="id >= 1000 and id < 1300", entity="id", time="year", freq="Y")
    sel = got.meta.extra["loads"][-1]["selection"]
    # pyarrow's own pruning from the footer's statistics, asked directly
    fragment = next(pads.dataset(str(path)).get_fragments())
    kept = fragment.split_by_row_group(filter=(pads.field("id") >= 1000) & (pads.field("id") < 1300))
    assert sel["row_groups_in_file"] == pq.ParquetFile(path).metadata.num_row_groups == 20
    assert sel["row_groups_read"] == len(kept) == 3  # ids 1000-1299 are rows 20,000-25,999: row groups 6, 7 and 8 of 3,000
    assert sel["rows_read"] == 9_000 and sel["rows_in_file"] == 60_000 and sel["rows_kept"] == 6_000
    assert "id >= 1000" in sel["pushed_down"] and sel["where"] == "id >= 1000 and id < 1300"


def test_what_is_not_pushed_is_said(panel_file):
    path, _ = panel_file
    got = Dataset.read(str(path), where="id < 100 and log(abs(x)) > 0 and not (y > 0)", entity="id", time="year", freq="Y")
    sel = got.meta.extra["loads"][-1]["selection"]
    assert sel["applied_after_reading"] == ["log(abs(x)) > 0", "not y > 0"]
    assert sel["row_groups_read"] == 1
    got = Dataset.read(str(path), where="x > y", entity="id", time="year", freq="Y")
    sel = got.meta.extra["loads"][-1]["selection"]
    assert "pushed_down" not in sel and sel["rows_read"] == 60_000


def test_pushdown_expressions():
    schema = pa.schema([("x", pa.float64()), ("n", pa.int32()), ("s", pa.string()), ("d", pa.date32())])
    expr, rest = condition_expression("x > 1 and 3 >= n and s in ['a'] and d < '2001-01-01' and n > 2.5", schema)
    assert str(expr) == ('((((x > 1) and (n <= 3)) and is_in(s, {value_set=string:[\n  "a"\n], null_matching_behavior=MATCH}))'
                         ' and (d < 2001-01-01))')
    assert rest == ["n > 2.5"]  # 2.5 against whole numbers: left to pandas
    expr, rest = condition_expression("s != 'a' or n == 1e12", schema)  # 1e12 does not fit in 32 bits
    assert expr is None and rest == ["s != 'a' or n == 1000000000000.0"]
    expr, _ = condition_expression("s != 'a'", schema)
    assert "is_null(s" in str(expr)  # pandas keeps missing values for !=
    expr, rest = condition_expression("n > 'a' and x > 'b'", schema)  # types that do not agree
    assert expr is None and len(rest) == 2


def test_the_condition_is_checked_first(panel_file):
    path, _ = panel_file
    with pytest.raises(SpecificationError, match="not allowed"):
        Dataset.read(str(path), where="x.__class__ == 1", entity="id", time="year", freq="Y")
    with pytest.raises(DataError, match="no column z"):
        Dataset.read(str(path), columns=["z"], entity="id", time="year", freq="Y")
    with pytest.raises(DataError, match="time axis"):
        Dataset.read(str(path), span=("1995", "1999"), entity="id")


def test_only_parquet_takes_a_selection(tmp_path):
    csv = tmp_path / "d.csv"
    pd.DataFrame({"t": ["2001Q1"], "x": [1.0]}).to_csv(csv, index=False)
    with pytest.raises(DataError, match="Parquet"):
        Dataset.read(str(csv), columns=["x"], time="t", freq="Q")


def test_a_pymetrix_file_with_its_time_axis_flags_and_metadata(tmp_path):
    rng = np.random.default_rng(1)
    idx = pd.period_range("1980Q1", periods=160, freq="Q")
    frame = pd.DataFrame({"a": rng.normal(size=160), "b": rng.normal(size=160), "c": rng.normal(size=160)},
                         index=pd.Index(idx, name="time"))
    base = Dataset.from_frame(frame)
    flags = pd.DataFrame({"a": None, "b": None}, index=base.frame.index, dtype=object)
    flags.loc[pd.Period("1990Q1", "Q"), "a"] = "e"  # read: inside the span
    flags.loc[pd.Period("2005Q2", "Q"), "b"] = "p"  # not read: outside the span, and b is not a column asked for
    data = Dataset(base.frame, base.axes, base.variables, flags, base.meta)
    path = tmp_path / "q.parquet"
    data.write(str(path), row_group_size=40)
    got = Dataset.read(str(path), columns=["a"], span=("1990q1", "1999Q4"), where="time < '1995Q3' or time > '1995Q3'")
    whole = Dataset.read(str(path)).select(span=("1990q1", "1999Q4"), where="time < '1995Q3' or time > '1995Q3'")
    pd.testing.assert_frame_equal(got.frame, whole.frame[["a"]])
    assert got.time_axis.freq == data.time_axis.freq
    sel = got.meta.extra["loads"][-1]["selection"]
    assert sel["row_groups_read"] == 1 and sel["rows_kept"] == 39  # 1990Q1-1999Q4 is rows 40-79: one row group
    assert list(got.flags.columns) == ["a"] and got.flags.loc[pd.Period("1990Q1", "Q"), "a"] == "e"


def test_a_partitioned_folder(tmp_path):
    df = _panel(30_000, seed=2)
    df["region"] = np.where(df.country == "DEU", "north", "south")
    folder = tmp_path / "parts"
    pq.write_to_dataset(pa.Table.from_pandas(df, preserve_index=False), folder, partition_cols=["year", "country"])
    files = len(list(folder.rglob("*.parquet")))
    assert files == 60
    where = "year >= '2005' and country == 'DEU' and x > 0"
    got = Dataset.read(str(folder), where=where, entity="id", time="year", freq="Y")
    sel = got.meta.extra["loads"][-1]["selection"]
    assert sel["files"] == 60 and sel["files_read"] == 5  # 2005-2009 for DEU: the folders' names rule the rest out
    # pandas: every file read, then selected (the partition columns come back as categories: made plain first)
    whole = pd.read_parquet(folder)
    whole["year"] = whole["year"].astype(int)
    whole["country"] = whole["country"].astype(str)
    expected = Dataset.from_frame(whole, entity="id", time="year", freq="Y").select(where=where).frame
    got_frame = got.frame.copy()
    got_frame["country"] = got_frame["country"].astype(str)
    pd.testing.assert_frame_equal(got_frame[expected.columns], expected, check_dtype=False)
    assert set(got.names) == set(expected.columns)
    # a scan of the folder: partition columns from the folder names
    s = scan(folder)
    assert s.files == 60 and s.rows == 30_000 and s.partitioning == ("year", "country")
    t = s.table()
    assert t.loc["year", "min"] == 1990 and t.loc["year", "max"] == 2009 and bool(t.loc["country", "partition"])


def test_scan_reads_metadata_only(panel_file):
    path, df = panel_file
    s = scan(path)
    assert s.rows == 60_000 and s.row_groups == 20 and s.files == 1
    t = s.table()
    assert t.loc["x", "min"] == df.x.min() and t.loc["x", "max"] == df.x.max()
    assert t.loc["y", "missing"] == df.y.isna().sum()
    assert t.loc["country", "min"] == "DEU" and t.loc["country", "max"] == "ITA"
    assert "60,000 rows" in str(s) and not s.large


def test_a_large_file_read_with_a_selection_is_quick(tmp_path):
    n = 2_000_000
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"id": np.arange(n) // 20, "year": 1990 + np.arange(n) % 20, "x": rng.normal(size=n),
                       "y": rng.normal(size=n)})
    path = tmp_path / "large.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, row_group_size=100_000)
    started = time.perf_counter()
    got = Dataset.read(str(path), columns=["x"], where="id < 5000 and x > 0", entity="id", time="year", freq="Y")
    seconds = time.perf_counter() - started
    sel = got.meta.extra["loads"][-1]["selection"]
    assert sel["row_groups_read"] == 1 and sel["rows_in_file"] == n
    mask = (df.id < 5000) & (df.x > 0)
    assert len(got.frame) == mask.sum()
    assert np.array_equal(got.frame["x"].to_numpy(), df.loc[mask].set_index(["id", "year"]).sort_index()["x"].to_numpy())
    assert seconds < 5


def test_fingerprint_follows_the_contents(tmp_path):
    path = tmp_path / "f.parquet"
    pq.write_table(pa.table({"x": [1.0, 2.0]}), path)
    first = fingerprint(path)
    assert fingerprint(path) == first
    pq.write_table(pa.table({"x": [1.0, 3.0]}), path)
    assert fingerprint(path) != first


def _session_with(path, **kw):
    from pymetrix.commands import LoadData, Session

    s = Session()
    s.run(LoadData(str(path), time="year", entity="id", freq="Y", name="big", **kw))
    return s


def test_a_project_embeds_the_selection_as_read(panel_file, tmp_path):
    from pymetrix.commands import load_project, save_project

    path, _ = panel_file
    s = _session_with(path, columns="x, y", where="id < 200")
    assert s["big"].names == ["x", "y"] and len(s["big"].frame) == 4_000
    project = save_project(s, tmp_path / "p.pymx")
    with zipfile.ZipFile(project) as z:
        embedded = pq.read_table(z.open("data/big.parquet"))
    assert embedded.num_rows == 4_000 and set(embedded.column_names) == {"id", "year", "x", "y"}
    opened = load_project(project)
    pd.testing.assert_frame_equal(opened["big"].frame, s["big"].frame)
    assert opened["big"].meta.extra["loads"][-1]["selection"]["rows_kept"] == 4_000


def test_a_project_can_keep_a_reference_and_refuses_a_changed_file(tmp_path):
    from pymetrix.commands import load_project, save_project

    df = _panel(6_000, seed=4)
    path = tmp_path / "ref.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, row_group_size=1_000)
    s = _session_with(path, where="id < 100", embed=False)
    assert s["big"].meta.vintage["embed"] is False
    project = save_project(s, tmp_path / "r.pymx")
    with zipfile.ZipFile(project) as z:
        assert "data/big.parquet" not in z.namelist()
        import json

        manifest = json.loads(z.read("manifest.json"))
    assert manifest["references"]["big"]["fingerprint"] == fingerprint(path)
    opened = load_project(project)  # read again from the file, with the same selection
    pd.testing.assert_frame_equal(opened["big"].frame, s["big"].frame)
    df.loc[0, "x"] = 99.0
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, row_group_size=1_000)
    with pytest.raises(DataError, match="has changed"):
        load_project(project)
    changed = load_project(project, changed_data="read")
    assert changed["big"].frame["x"].iloc[0] == 99.0
    path.unlink()
    with pytest.raises(DataError, match="does not exist"):
        load_project(project)


def test_load_data_shows_the_selection_for_large_parquet_only(panel_file, tmp_path, monkeypatch):
    from pymetrix.commands import LoadData
    from pymetrix.forms import form_spec
    from pymetrix.io import selection

    path, _ = panel_file
    specs = {s.name: s for s in form_spec(LoadData)}
    for name in ("columns", "where", "start", "end", "embed"):
        assert not specs[name].visible({"path": str(path)})  # 1 MB: below the threshold
    monkeypatch.setattr(selection, "LARGE", 1000)
    for name in ("columns", "where", "start", "end", "embed"):
        assert specs[name].visible({"path": str(path)})
    csv = tmp_path / "big.csv"
    csv.write_text("a\n" + "1\n" * 2000)
    assert not specs["where"].visible({"path": str(csv)})
    # a CSV keeps no selection, even one left in the hidden fields
    cmd = LoadData(str(csv), where="a > 0", columns="a")
    assert "where" not in cmd.to_code() and "columns" not in cmd.to_code()
    cmd = LoadData(str(path), time="year", entity="id", freq="Y", columns="x", where="id < 10", start="1995", end="1996")
    assert "columns=['x']" in cmd.to_code() and "where='id < 10'" in cmd.to_code() and "span=('1995', '1996')" in cmd.to_code()
    assert LoadData(str(path), where="x.y > 1").validate(None)


def test_the_load_data_dialog_offers_columns_for_a_large_file(panel_file, monkeypatch):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.commands import LoadData, Session
    from pymetrix.io import selection
    from pymetrix.ui.qt import command_dialog

    path, _ = panel_file
    dialog = command_dialog(LoadData, Session())
    dialog.editors["path"].set(str(path))
    dialog.refresh_preview()
    assert not dialog.editors["where"].widget.isVisibleTo(dialog)  # small: read whole
    monkeypatch.setattr(selection, "LARGE", 1000)
    dialog = command_dialog(LoadData, Session())
    dialog.editors["path"].set(str(path))
    dialog.refresh_preview()
    assert dialog.editors["where"].widget.isVisibleTo(dialog)
    columns = dialog.editors["columns"].widget
    assert [columns.item(i).text() for i in range(columns.count())] == ["id", "year", "x", "country", "y", "flag", "day"]
