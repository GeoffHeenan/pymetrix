"""pymetrix-psets against a stand-in for PSETS's plug-in API (``psets`` 1.0: the same calls and records), so the
plugin is tested where PSETS is not installed, as in pymetrix's CI. ``test_with_psets.py`` runs it against PSETS
itself."""

import importlib.util
import sys
import types
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, backtest
from pymetrix.commands import Session, load_project, save_project
from pymetrix.io import registry as io_registry
from pymetrix.metadata import Adjustment

PLUGIN_SRC = __import__("pathlib").Path(__file__).parents[1] / "src"


# -- a stand-in for psets 1.0 -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Series:  # psets.Series, the fields this plugin reads
    series_id: str
    title: str
    freq: str
    unit: str = ""
    multiplier: float = 1.0
    source: str = ""
    source_name: str = ""
    dataset: str = ""
    dataset_name: str = ""
    sector: str = ""
    topic: str = ""
    geo: str = ""
    seasonal_adj: str = ""
    first_period: str = ""
    last_period: str = ""
    last_loaded_at: str = ""
    fiscal_year_end: str = ""
    notes: str = ""

    @property
    def span(self):
        return (self.first_period, self.last_period)


@dataclass(frozen=True)
class Segment:
    series_id: str
    role: str
    rank: int = 0
    link_period: str = ""
    link_factor: float | None = None
    provider: str = ""
    provider_code: str = ""


@dataclass(frozen=True)
class Provenance:
    code: str
    canonical: bool
    segments: tuple = ()


@dataclass(frozen=True)
class Issue:
    issue_id: str
    publisher: str
    summary: str
    kind: str
    status: str

    @property
    def open(self):
        return self.status != "resolved"


def _quarter_ends(first: str, n: int) -> list[str]:
    return [p.end_time.date().isoformat() for p in pd.period_range(first, periods=n, freq="Q")]


SERIES = {
    "NGDP_R.Q": Series("NGDP_R.Q", "Real GDP", "Q", unit="NLK", multiplier=1e-06, source="NSO", source_name="Norland Statistics Office",
                       seasonal_adj="NSA", geo="XNL", first_period="2010-03-31", last_period="2019-12-31"),
    "NGDP.Q": Series("NGDP.Q", "Nominal GDP", "Q", unit="NLK", source="NSO", source_name="Norland Statistics Office",
                     seasonal_adj="SA", first_period="2010-03-31", last_period="2019-12-31"),
    "PCPI_LOC.M": Series("PCPI_LOC.M", "Local CPI", "M", unit="index", source="NSO"),
}
# two releases of real GDP: the 2015-06-15 release covers 2010Q1-2017Q4 (a forecast-era extract); the 2020-03-15 one
# revises every period and runs on to 2019Q4
PERIODS = _quarter_ends("2010Q1", 40)
OBS = ([("NGDP_R.Q", p, 100.0 + i, "2015-06-15") for i, p in enumerate(PERIODS[:32])]
       + [("NGDP_R.Q", p, 101.0 + i, "2020-03-15") for i, p in enumerate(PERIODS)]
       + [("NGDP.Q", p, 150.0 + 2 * i, "2020-03-15") for i, p in enumerate(PERIODS)])


class FakeDatabase:
    def __init__(self, alias):
        self.alias, self.path = alias, f"/data/{alias or 'default'}.sqlite"

    def series(self, code):
        if code not in SERIES:
            raise KeyError(f"No series {code!r}")  # psets.UnknownSeries is a KeyError
        return SERIES[code]

    def frame(self, codes, *, freq="", agg="", start="", end="", as_of="", transform="", long=False):
        rows = pd.DataFrame(OBS, columns=["series_id", "period", "value", "vintage"])
        rows = rows[rows["series_id"].isin(codes)]
        if as_of:
            rows = rows[rows["vintage"] <= as_of]
        rows = rows[(rows["period"] >= (start or "")) & (rows["period"] <= (end or "9999"))]
        rows = rows.sort_values("vintage").groupby(["series_id", "period"], as_index=False).last()
        return rows.reset_index(drop=True)

    def vintages(self, code):
        return sorted({v for c, _, _, v in OBS if c == code})

    def provenance(self, code):
        if code == "NGDP_R.Q":
            return Provenance(code, True, (Segment("NSO.NA.REALGDP", "primary", 1, provider="NSO", provider_code="REALGDPPP"),))
        return Provenance(code, False)

    def issues(self, code=""):
        return [Issue("I-7", "NSO", "2019Q4 printed twice", "duplicate", "open")] if code == "NGDP.Q" else []

    def search(self, text="", **facets):
        hits = [s for s in SERIES.values() if text.lower() in s.title.lower() and all(getattr(s, k) == v for k, v in facets.items())]
        return pd.DataFrame({"series_id": [s.series_id for s in hits]})

    def facets(self, facet="", **criteria):
        return {"freq": [("Q", 2), ("M", 1)]}


@pytest.fixture
def psets_plugin(monkeypatch):
    fake = types.ModuleType("psets")
    fake.__version__ = "1.0"
    fake.open = lambda database="": FakeDatabase(database)
    monkeypatch.setitem(sys.modules, "psets", fake)
    monkeypatch.syspath_prepend(str(PLUGIN_SRC))
    saved = dict(io_registry._READERS)
    spec = importlib.util.spec_from_file_location("pymetrix_psets", PLUGIN_SRC / "pymetrix_psets" / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "pymetrix_psets", module)
    spec.loader.exec_module(module)
    module.register()  # what the pymetrix.io entry point does
    yield module
    io_registry._READERS.clear()
    io_registry._READERS.update(saved)


def test_reads_series_with_their_metadata(psets_plugin):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", psets_plugin.PsetsIssueWarning)
        data = Dataset.read("psets://forecasts/NGDP_R.Q,NGDP.Q")
    assert data.names == ["ngdp_r_q", "ngdp_q"] and str(data.frame.index[0]) == "2010Q1"
    assert data["ngdp_r_q"].iloc[0] == 101.0  # the latest release
    real = data.variables["ngdp_r_q"]
    assert real.label == "Real GDP" and real.units == "NLK" and real.source_key == "NGDP_R.Q"
    assert real.source == "PSETS forecasts: Norland Statistics Office (NGDP_R.Q)"
    assert real.vintage.released.isoformat() == "2020-03-15"  # the publisher's release, not the load date
    assert real.external_ref == "psets://forecasts/NGDP_R.Q@2020-03-15"
    assert real.extra["psets_multiplier"] == 1e-06 and real.extra["psets_provenance"][0]["role"] == "primary"
    assert data.variables["ngdp_q"].adjustment is Adjustment.SA
    assert data.meta.vintage["policy"] == "latest"


def test_a_vintage_is_the_data_as_released_then(psets_plugin):
    old = Dataset.read("psets://forecasts/NGDP_R.Q", vintage="2016-01-01")
    assert len(old.frame) == 32 and old["ngdp_r_q"].iloc[0] == 100.0
    assert old.variables["ngdp_r_q"].vintage.released.isoformat() == "2015-06-15"
    assert old.meta.vintage == {"policy": "as_of", "date": "2016-01-01"}


def test_known_issues_are_raised_where_they_would_stop_someone(psets_plugin):
    with pytest.warns(psets_plugin.PsetsIssueWarning, match="2019Q4 printed twice"):
        data = Dataset.read("psets://forecasts/NGDP.Q")
    assert data.variables["ngdp_q"].extra["psets_issues"][0]["id"] == "I-7"


def test_bad_addresses_and_mixed_frequencies_are_refused(psets_plugin):
    from pymetrix.errors import SpecificationError

    with pytest.raises(SpecificationError, match="different frequencies"):
        Dataset.read("psets://forecasts/NGDP_R.Q,PCPI_LOC.M")
    with pytest.raises(SpecificationError, match="take start, end, freq, agg"):
        Dataset.read("psets://forecasts/NGDP_R.Q?colour=red")
    with pytest.raises(KeyError):
        Dataset.read("psets://forecasts/NOPE.Q")
    sys.modules["psets"].__version__ = "2.0"
    with pytest.raises(SpecificationError, match="needs PSETS plug-in API 1.x"):
        Dataset.read("psets://forecasts/NGDP_R.Q")


def test_a_project_records_the_read_and_keeps_the_reference(psets_plugin, tmp_path):
    s = Session()
    s.run_code("data = Dataset.read('psets://forecasts/NGDP_R.Q', vintage='2016-01-01')")
    record = s.history[-1]
    assert type(record.command).__name__ == "Call"
    assert record.code == "data = Dataset.read('psets://forecasts/NGDP_R.Q', vintage='2016-01-01')"
    opened = load_project(save_project(s, tmp_path / "psets.pymx"), trust=False)  # a call: no typed code to trust
    assert opened["data"].variables["ngdp_r_q"].external_ref == "psets://forecasts/NGDP_R.Q@2015-06-15"
    assert np.allclose(opened["data"].frame.to_numpy(), s["data"].frame.to_numpy())


def test_vintages_for_a_backtest(psets_plugin):
    origins = pd.period_range("2015Q4", "2017Q4", freq="Q")
    vintages = psets_plugin.psets_vintages("psets://forecasts/NGDP_R.Q", origins)
    assert vintages[pd.Period("2015Q4")]["ngdp_r_q"].iloc[0] == 100.0  # only the 2015 release was out then
    latest = Dataset.read("psets://forecasts/NGDP_R.Q")
    fit = Equation("ngdp_r_q ~ ngdp_r_q[-1]").fit(latest)
    result = backtest(fit, latest, ("2015Q4", "2017Q4"), horizon=2, vintages=vintages)
    assert len(result.errors) > 0


def test_the_catalogue_answers_the_get_data_questions(psets_plugin):
    catalogue = psets_plugin.PsetsCatalogue("forecasts")
    found = catalogue.search("gdp", {"freq": "Q"})
    assert [v.source_key for v in found] == ["NGDP_R.Q", "NGDP.Q"] and found[0].label == "Real GDP"
    assert catalogue.facets() == {"freq": [("Q", 2), ("M", 1)]}
    assert catalogue.describe(["NGDP_R.Q"])[0].vintage.released.isoformat() == "2020-03-15"
    assert catalogue.address(["NGDP_R.Q", "NGDP.Q"]) == "psets://forecasts/NGDP_R.Q,NGDP.Q"
    assert psets_plugin._fiscal_month("--06-30") == "JUN" and psets_plugin._fiscal_month("July") == "JUL"
