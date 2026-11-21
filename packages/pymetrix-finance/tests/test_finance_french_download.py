"""The Fama-French example is downloaded from Kenneth R. French's data library on first use, not shipped.

Checked against what, with no network (conftest.py serves the simulated stand-in in the library's file layout):
- the dataset built from the five files equals the stand-in's simulated returns, cut to 1949-01..2017-03 and divided
  by 100, exactly; its columns, labels and licence (the library's copyright, not redistributed);
- the first load fetches the five files once and caches the dataset in the user's cache folder; later loads read
  the cache and fetch nothing;
- offline, the first load fails with a DataError that names the library and the cache path, and leaves no file;
- the library's missing values (-99.99, -999) read as missing; a file whose layout changed is refused;
- the plugin ships no copy of the data (no french.parquet in the package).
"""

import io
import urllib.error
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset
from pymetrix.errors import DataError
from pymetrix_finance.datasets import french

import french_stand_in


def test_the_first_load_downloads_and_caches(french_offline):
    assert not french.cache_path().exists()
    data = Dataset.example("french")
    assert sorted(french_offline) == sorted(f"{french.BASE}{stem}_CSV.zip" for stem in french.FILES)
    assert french.cache_path().is_file()
    again = Dataset.example("french")
    assert len(french_offline) == 5  # read from the cache: nothing fetched the second time
    pd.testing.assert_frame_equal(again.frame, data.frame)


def test_the_dataset_is_the_library_s_returns_as_fractions(french_offline):
    data = Dataset.example("french")
    frame = data.frame
    assert list(frame.columns) == french.COLUMNS and len(french.COLUMNS) == 35
    assert str(frame.index[0]) == "1949-01" and str(frame.index[-1]) == "2017-03" and len(frame) == 819
    simulated = french_stand_in.returns().round(2).loc["1949-01":"2017-03"]
    names = {library: ours for columns in french.FILES.values() for library, ours in columns.items()}
    expected = (simulated[list(names)].rename(columns=names)[french.COLUMNS] / 100).to_numpy()
    assert np.array_equal(frame.to_numpy(), expected)
    assert data.variables["hlth"].label == "industry: health care" and data.variables["s5m1"].label == "size 5, prior return 1"
    assert "copyright Eugene F. Fama and Kenneth R. French" in data.meta.license and "not redistributed" in data.meta.license
    assert data.meta.title == french.TITLE


def test_offline_the_error_says_what_to_do(french_offline, monkeypatch):
    def unreachable(url):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(french, "_get", unreachable)
    with pytest.raises(DataError, match=r"downloaded from Kenneth R\. French's data library on first use") as caught:
        Dataset.example("french")
    assert "no route to host" in str(caught.value) and str(french.cache_path()) in str(caught.value)
    folder = french.cache_path().parent
    assert not folder.exists() or not any(folder.iterdir())  # nothing half-written left in the cache


def test_missing_values_and_a_changed_layout():
    text = "notes\n\n,A,B\n194901, 1.00,-99.99\n194902,-999, 2.50\n\n  Annual\n,A,B\n1949, 1.0, 2.0\n"
    table = french.monthly_table(text)
    assert [str(p) for p in table.index] == ["1949-01", "1949-02"]
    assert table["A"].iloc[0] == 1.0 and np.isnan(table["B"].iloc[0]) and np.isnan(table["A"].iloc[1]) and table["B"].iloc[1] == 2.5
    served = french_stand_in.library()
    stem = "F-F_Momentum_Factor"
    url = f"{french.BASE}{stem}_CSV.zip"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{stem}.csv", "notes\n\n,Momentum\n194901, 1.0\n")
    served[url] = buffer.getvalue()
    with pytest.raises(DataError, match=r"has no column Mom: the data library's layout has changed"):
        french.build(served.__getitem__)
    served[url] = b"<html>not a zip</html>"
    with pytest.raises(DataError, match="in a form pymetrix does not read"):
        french.build(served.__getitem__)


def test_the_plugin_ships_no_copy():
    package = Path(french.__file__).parent
    assert not (package / "french.parquet").exists()
    assert "downloaded on first use" in repr(french.Download()) and french.Download.source == french.LIBRARY
