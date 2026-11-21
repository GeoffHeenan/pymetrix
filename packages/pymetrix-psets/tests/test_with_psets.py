"""pymetrix-psets against PSETS itself: a real database, built with PSETS's own migrations, opened by a registry
alias through ``psets.open``. Skipped where PSETS is not installed (it is not on PyPI; see the README).

The database has the shapes that matter: a series revised by a later release, a period the publisher left
missing, and a monthly series PSETS can convert to quarters as it reads."""

import sys
from datetime import date
from pathlib import Path

import pytest

psets = pytest.importorskip("psets")
try:
    from psets import db as psets_db
    from psets import migrate as psets_migrate
except ImportError:  # PSETS before everything moved under the psets namespace
    pytest.skip("this PSETS predates the psets namespace (psets.db, psets.migrate)", allow_module_level=True)

from pymetrix import Dataset
from pymetrix.errors import SpecificationError
from pymetrix.io import registry as io_registry
from pymetrix.metadata import Adjustment

SRC = Path(__file__).parents[1] / "src"


@pytest.fixture
def forecasts(tmp_path, monkeypatch):
    """A PSETS database registered as the alias ``forecasts`` in a registry of its own, through a settings file
    that only this test reads (``PSETS_CONFIG``)."""
    path = tmp_path / "forecasts.sqlite"
    psets_migrate.migrate(str(path))
    with psets_db.connect(str(path)) as con:
        con.execute("INSERT INTO sources (source_id, name) VALUES ('NSO', 'Norland Statistics Office')")
        con.execute("INSERT INTO datasets (dataset_id, source_id, name) VALUES ('NSO_CPI', 'NSO', 'Prices')")
        con.execute("INSERT INTO series (series_id, dataset_id, source_id, title, freq, unit, seasonal_adj, multiplier, "
                    "decimals) VALUES ('CPI.M', 'NSO_CPI', 'NSO', 'Consumer prices, all items', 'M', 'Index', 'NSA', 1.0, 2)")
        months = [("2024-01-31", 100.0), ("2024-02-29", 101.0), ("2024-03-31", 102.0),
                  ("2024-04-30", None), ("2024-05-31", 104.0), ("2024-06-30", 105.0)]
        for period, value in months:
            con.execute("INSERT INTO observations (series_id, period, vintage, value) VALUES ('CPI.M', ?, '2024-07-15', ?)",
                        (period, value))
        con.execute("INSERT INTO observations (series_id, period, vintage, value) "
                    "VALUES ('CPI.M', '2024-01-31', '2024-08-15', 100.5)")  # January revised by a later release
    (tmp_path / "databases.ini").write_text("[forecasts]\npath = forecasts.sqlite\naccess = standard\n", encoding="utf-8")
    (tmp_path / "config").write_text(f"registry = {tmp_path / 'databases.ini'}\n", encoding="utf-8")
    monkeypatch.setenv("PSETS_CONFIG", str(tmp_path / "config"))
    return path


@pytest.fixture
def plugin(monkeypatch):
    monkeypatch.syspath_prepend(str(SRC))
    sys.modules.pop("pymetrix_psets", None)  # not a stand-in-backed copy from the other tests
    import pymetrix_psets

    saved = dict(io_registry._READERS)
    pymetrix_psets.register()  # what the pymetrix.io entry point does
    yield pymetrix_psets
    io_registry._READERS.clear()
    io_registry._READERS.update(saved)
    sys.modules.pop("pymetrix_psets", None)


def test_reads_the_latest_figures_with_the_gap_kept(forecasts, plugin):
    data = Dataset.read("psets://forecasts/CPI.M")
    values = data.frame["cpi_m"]
    assert [str(p) for p in values.index] == ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"]
    assert values.iloc[0] == 100.5  # the revised January
    assert values.isna().tolist() == [False, False, False, True, False, False]  # April: published as missing


def test_the_series_carries_its_psets_metadata(forecasts, plugin):
    var = Dataset.read("psets://forecasts/CPI.M").variables["cpi_m"]
    assert (var.label, var.units, var.adjustment) == ("Consumer prices, all items", "Index", Adjustment.NSA)
    assert var.source_key == "CPI.M" and "Norland Statistics Office" in var.source
    assert var.vintage.released == date(2024, 8, 15)  # the publisher's release, never the load date
    assert var.external_ref == "psets://forecasts/CPI.M@2024-08-15"
    assert var.extra["psets_freq"] == "M" and var.extra["psets_dataset"] == "Prices"


def test_a_vintage_gives_the_figures_as_then_released(forecasts, plugin):
    then = Dataset.read("psets://forecasts/CPI.M", vintage="2024-07-31")
    assert then.frame["cpi_m"].iloc[0] == 100.0  # before January's revision
    assert then.variables["cpi_m"].vintage.released == date(2024, 7, 15)


def test_psets_converts_downward_as_it_reads(forecasts, plugin):
    quarters = Dataset.read("psets://forecasts/CPI.M?freq=Q&agg=avg").frame["cpi_m"]
    assert [str(p) for p in quarters.index] == ["2024Q1", "2024Q2"]
    assert quarters.iloc[0] == pytest.approx((100.5 + 101.0 + 102.0) / 3)


def test_a_code_psets_does_not_hold_is_an_error_not_an_empty_series(forecasts, plugin):
    with pytest.raises(psets.UnknownSeries):
        Dataset.read("psets://forecasts/NOT_THERE.M")


def test_an_address_must_name_codes(forecasts, plugin):
    with pytest.raises(SpecificationError, match="series codes"):
        Dataset.read("psets://forecasts/")
