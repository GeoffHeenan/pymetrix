"""Data connectors (FRED, World Bank, SDMX), from recorded responses: the tests never use the network."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset
from pymetrix.ext import connectors
from pymetrix.plugins import plugins

FIXTURES = Path(__file__).parent / "fixtures" / "connectors"


@pytest.fixture()
def responses(monkeypatch):
    """Answer each URL from a fixture file; record the URLs asked for."""
    asked: list[str] = []
    routes = {
        "fredgraph.csv?id=GDPC1&vintage_date": "fred_gdpc1_2020.csv",
        "fredgraph.csv?id=GDPC1": "fred_gdpc1.csv",
        "fredgraph.csv?id=UNRATE": "fred_unrate.csv",
        "worldbank.org/v2/country/WSM;TON/indicator/NY.GDP.MKTP.CD": None,  # by page
        "data-api.ecb.europa.eu": "ecb_exr.csv",
        "ec.europa.eu/eurostat": "estat_hicp.csv",
        "stats.bis.org/api/v1/data/WS_CBPOL/M.US": "bis_cbpol.csv",
        "stats.bis.org/api/v1/data/WS_CBPOL/Q.US": "sdmx_error.xml",
        "api.imf.org": "imf_cpi.xml",
        "data.mycb.org": "ecb_exr.csv",
    }

    def fake(url: str) -> bytes:
        asked.append(url)
        for pattern, name in routes.items():
            if pattern in url:
                if name is None:
                    name = "wb_page2.json" if "page=2" in url else "wb_page1.json"
                return (FIXTURES / name).read_bytes()
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(connectors, "_get", fake)
    return asked


def test_fred(responses):
    data = Dataset.read("fred://GDPC1")
    assert data.kind == "time_series" and data.frame.index.freqstr.startswith("Q") and data.names == ["gdpc1"]
    assert str(data.frame.index[0]) == "1947Q1" and data["gdpc1"].iloc[0] == pytest.approx(2182.681)
    assert "Federal Reserve Bank of St. Louis" in data.variables["gdpc1"].source
    assert data.variables["gdpc1"].vintage is None  # the CSV download does not say when the data were released
    old = Dataset.read("fred://GDPC1?vintage=2020-01-30")
    assert "vintage_date=2020-01-30" in responses[-1] and "ALFRED vintage as of 2020-01-30" in old.variables["gdpc1"].source
    with pytest.raises(ValueError, match="different frequencies"):
        Dataset.read("fred://GDPC1,UNRATE")
    monthly = Dataset.read("fred://UNRATE")
    assert monthly.frame.index.freqstr.startswith("M")


def test_world_bank_panel_with_its_release_date(responses):
    data = Dataset.read("worldbank://NY.GDP.MKTP.CD?country=WSM;TON&date=2021:2023&per_page=3")
    assert sum("page=" in u for u in responses) == 2  # followed the second page
    assert data.kind == "panel" and data.names == ["ny_gdp_mktp_cd"]
    assert sorted(data.frame.index.get_level_values("country").unique()) == ["TON", "WSM"]
    assert data.frame.index.get_level_values("time").freqstr.startswith("Y")
    var = data.variables["ny_gdp_mktp_cd"]
    assert var.label == "GDP (current US$)" and var.vintage.released == date(2026, 10, 8)  # the World Bank's last update
    assert len(data) == 6 and data["ny_gdp_mktp_cd"].notna().all()


def test_sdmx_csv_from_several_providers(responses):
    ecb = Dataset.read("sdmx://ECB/EXR/M.USD+GBP.EUR.SP00.A?startPeriod=2024-01&endPeriod=2024-03")
    assert "startPeriod=2024-01" in responses[-1] and "M.USD+GBP.EUR.SP00.A" in responses[-1]
    assert ecb.names == ["gbp", "usd"] or ecb.names == ["usd", "gbp"]  # named by the dimension that differs
    assert ecb.frame.index.freqstr.startswith("M") and len(ecb) == 3
    assert "US dollar/Euro" in ecb.variables["usd"].label
    estat = Dataset.read("sdmx://EUROSTAT/prc_hicp_manr/M.RCH_A.CP00.EA", name="hicp")
    assert estat.names == ["hicp"] and estat.variables["hicp"].vintage.released == date(2026, 2, 6)  # LAST UPDATE 06/02/26
    bis = Dataset.read("sdmx://BIS/WS_CBPOL/M.US")
    assert bis["ws_cbpol"].iloc[0] == pytest.approx(5.375)
    with pytest.raises(ValueError, match="No data for data query"):
        Dataset.read("sdmx://BIS/WS_CBPOL/Q.US")


def test_sdmx_xml_and_panels_by_a_dimension(responses):
    imf = Dataset.read("sdmx://IMF/IMF.STA,CPI/USA+WSM.CPI.CP01.IX.M")
    assert set(imf.names) == {"usa", "wsm"} and imf.frame.index.freqstr.startswith("M")  # attributes that differ are not names
    assert np.isnan(imf["wsm"][pd.Period("2024-02", "M")]) and imf["usa"].iloc[-1] == pytest.approx(141.9)
    assert imf.variables["usa"].vintage.released == date(2026, 10, 11)  # UPDATE_DATE
    panel = Dataset.read("sdmx://IMF/IMF.STA,CPI/USA+WSM.CPI.CP01.IX.M?entity=COUNTRY")
    assert panel.kind == "panel" and panel.names == ["cpi"] and panel.variables["cpi"].label == "*.CPI.CP01.IX.M"
    assert panel["cpi"].loc[("WSM", pd.Period("2024-03", "M"))] == pytest.approx(181.0)
    with pytest.raises(ValueError, match="no dimension 'REGION'"):
        Dataset.read("sdmx://IMF/IMF.STA,CPI/USA+WSM.CPI.CP01.IX.M?entity=REGION")


def test_institutions_add_their_own_sdmx_service(responses):
    with pytest.raises(ValueError, match="unknown SDMX provider 'MYCB'"):
        Dataset.read("sdmx://MYCB/EXR/M.USD.EUR.SP00.A")
    connectors.register_sdmx_provider("MYCB", "https://data.mycb.org/sdmx/{flow}/{key}?format=csvdata")
    try:
        data = Dataset.read("sdmx://MYCB/EXR/M.USD+GBP.EUR.SP00.A")
        assert responses[-1].startswith("https://data.mycb.org/sdmx/EXR/M.USD+GBP.EUR.SP00.A")
        assert len(data.names) == 2
    finally:
        connectors._PROVIDERS.pop("MYCB")
    with pytest.raises(ValueError, match="needs {flow} and {key}"):
        connectors.register_sdmx_provider("BAD", "https://example.org/data")


def test_the_plugin_is_listed():
    assert "pymetrix-connectors" in [p.name for p in plugins()]
    with pytest.raises(ValueError, match="sdmx://PROVIDER/FLOW/KEY"):
        Dataset.read("sdmx://ECB/EXR")


def test_every_connector_records_the_minimum_provenance(responses):
    """Source, the series' ID there, when pymetrix fetched it, and the release date where the provider gives one."""
    fred = Dataset.read("fred://GDPC1").variables["gdpc1"]
    assert fred.source_key == fred.source_id == "GDPC1" and fred.retrieved[:4].isdigit() and fred.vintage is None
    alfred = Dataset.read("fred://GDPC1?vintage=2020-01-30").variables["gdpc1"]
    assert alfred.vintage.released == date(2020, 1, 30)  # the data as released by that date
    wb = Dataset.read("worldbank://NY.GDP.MKTP.CD?country=WSM;TON&date=2021:2023&per_page=3").variables["ny_gdp_mktp_cd"]
    assert wb.source_key == "NY.GDP.MKTP.CD" and wb.retrieved
    ecb = Dataset.read("sdmx://ECB/EXR/M.USD+GBP.EUR.SP00.A?startPeriod=2024-01&endPeriod=2024-03")
    keys = sorted(v.source_key for v in ecb.variables.values())
    assert keys == ["EXR/M.GBP.EUR.SP00.A", "EXR/M.USD.EUR.SP00.A"] and all(v.retrieved for v in ecb.variables.values())


def test_files_record_their_name_and_no_release_date(tmp_path):
    from pymetrix.io.tabular import read_csv

    path = tmp_path / "rates.csv"
    pd.DataFrame({"time": ["2020Q1", "2020Q2"], "r": [1.0, 2.0]}).to_csv(path, index=False)
    var = read_csv(str(path), time="time", freq="Q").variables["r"]
    assert var.source == "file rates.csv" and var.retrieved and var.vintage is None  # a file has no release date


def test_an_external_reference_is_kept_saved_and_dropped_with_replaced_values(tmp_path):
    from dataclasses import replace

    data = Dataset.example("sample_macro")
    ref = replace(data.variables["y"], external_ref="psets://national-accounts/gdp/2026-09", source_key="GDP")
    data = Dataset(data.frame, data.axes, {**data.variables, "y": ref}, data.flags, data.meta)
    data.write(tmp_path / "d.parquet")
    back = Dataset.read(tmp_path / "d.parquet").variables["y"]
    assert back.external_ref == "psets://national-accounts/gdp/2026-09" and back.source_id == "GDP"
    assert data.with_series(y=data["y"] * 2).variables["y"].external_ref == ""
