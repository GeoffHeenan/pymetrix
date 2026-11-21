"""Searching and browsing data connectors (pymetrix.connectors), the cache of what was found, and File → Browse series….

Nothing here uses the network: every request is answered from tests/fixtures/connectors (the readers' recorded
responses) and tests/fixtures/connectors/search. The search fixtures are written by hand from the providers'
documented formats, since the providers could not be reached when they were made: FRED's web service JSON
(fred/series/search, fred/category/children, fred/category/series; the API needs a key, which these tests do not
have), the World Bank's v2 JSON ([header, rows] for /sources, /source/2/indicator, /topic, /country) and SDMX-ML 2.1
structure messages (an ECB dataflow list, the EXR dataflow with its data structure, concepts and codelists, an IMF
list whose flows belong to other agencies). The EXR structure's dimensions and codes are those of the ECB's EXR
series keys (M.USD.EUR.SP00.A), checked against the key the readers' tests use.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

from pymetrix import Dataset
from pymetrix import connectors as C
from pymetrix.commands import LoadFromConnector, Session
from pymetrix.errors import NotFoundError, SpecificationError
from pymetrix.ext import connectors as ext
from pymetrix.io import connectors as described

FIX = Path(__file__).parent / "fixtures" / "connectors"
SEARCH = FIX / "search"
KEY = "abcdef0123456789abcdef0123456789"  # FRED keys are 32 lower-case hex characters


def _page(url: str) -> int:
    found = re.search(r"[?&]page=(\d+)", url)
    return int(found.group(1)) if found else 1


ROUTES = [  # (a pattern in the URL, the fixture): the first that matches answers
    ("fred/series/search", SEARCH / "fred_series_search.json"),
    ("fred/category/children?category_id=0&", SEARCH / "fred_category_children_0.json"),
    ("fred/category/children?category_id=32447&", SEARCH / "fred_category_children_32447.json"),
    ("fred/category/series?category_id=32447&", SEARCH / "fred_category_series_32447.json"),
    ("fredgraph.csv?id=UNRATE", FIX / "fred_unrate.csv"),
    ("fredgraph.csv?id=GDPC1", FIX / "fred_gdpc1.csv"),
    ("worldbank.org/v2/sources?", SEARCH / "wb_sources.json"),
    ("worldbank.org/v2/source/2/indicator", lambda url: SEARCH / f"wb_source2_indicators_p{_page(url)}.json"),
    ("worldbank.org/v2/source/99/indicator", SEARCH / "wb_error.json"),
    ("worldbank.org/v2/topic?", SEARCH / "wb_topics.json"),
    ("worldbank.org/v2/topic/3/indicator", SEARCH / "wb_topic3_indicators.json"),
    ("worldbank.org/v2/country?", SEARCH / "wb_countries.json"),
    ("worldbank.org/v2/country/TON;WSM/indicator/NY.GDP.MKTP.CD", lambda url: FIX / f"wb_page{_page(url)}.json"),
    ("ecb.europa.eu/service/dataflow/ECB/EXR/latest?references=all", SEARCH / "ecb_exr_structure.xml"),
    ("ecb.europa.eu/service/dataflow/ECB/NOPE", SEARCH / "sdmx_no_flow.xml"),
    ("ecb.europa.eu/service/dataflow/ECB", SEARCH / "ecb_dataflows.xml"),
    ("api.imf.org/external/sdmx/2.1/dataflow/all", SEARCH / "imf_dataflows.xml"),
    ("data.mycb.org/rest/dataflow/MYCB", SEARCH / "ecb_dataflows.xml"),
    ("data-api.ecb.europa.eu/service/data/", FIX / "ecb_exr.csv"),
    ("/categoryscheme/", SEARCH / "sdmx_no_flow.xml"),  # no categories (an SDMX "No Results Found"): grouped by name
]


@pytest.fixture()
def net(monkeypatch, tmp_path):
    """Every request answered from a fixture (and counted); a fresh PYMETRIX_HOME, so a fresh cache; no FRED key."""
    monkeypatch.setenv("PYMETRIX_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    C.set_api_key("fred", None)
    asked: list[str] = []

    def fake(url: str) -> bytes:
        asked.append(url)
        for pattern, answer in ROUTES:
            if pattern in url:
                return Path(answer(url) if callable(answer) else answer).read_bytes()
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(ext, "_get", fake)
    yield asked
    C.set_api_key("fred", None)


# -- what each provider answers ------------------------------------------------------------------------------------------


def test_connector_descriptions_say_whether_they_can_be_searched():
    conns = described()
    assert all(conns[k].search and conns[k].search_note for k in ("fred", "worldbank", "sdmx"))
    assert "API key" in conns["fred"].search_note
    assert C.searchable() == sorted({"fred", "worldbank", "sdmx"} | set(C.searchable()))
    with pytest.raises(NotFoundError, match="cannot be searched"):
        C.search("nosuch", "gdp")


def test_fred_search_needs_a_key_and_reads_freds_json(net, monkeypatch):
    with pytest.raises(SpecificationError, match="FRED's search needs an API key"):
        C.search("fred", "unemployment")
    assert net == []  # nothing was asked without a key
    monkeypatch.setenv("FRED_API_KEY", KEY)  # the environment's key...
    found = C.search("fred", "unemployment rate", limit=3)
    assert "search_text=unemployment+rate" in net[-1] and f"api_key={KEY}" in net[-1] and "limit=3" in net[-1]
    assert [s.code for s in found] == ["UNRATE", "UNRATENSA", "NROU"]  # FRED's order (search rank), cut at the limit
    u = found[0]
    assert (u.title, u.frequency, u.units, u.start, u.end) == ("Unemployment Rate", "Monthly", "Percent", "1948-01-01", "2026-09-01")
    assert u.last_updated == "2026-10-03 07:46:02-05" and u.seasonal_adjustment == "Seasonally Adjusted"
    assert u.notes.startswith("The unemployment rate represents") and u.address == "fred://UNRATE"
    assert found[2].frequency == "Quarterly"
    frame = C.to_frame(found)
    assert list(frame["code"]) == ["UNRATE", "UNRATENSA", "NROU"] and frame.loc[0, "address"] == "fred://UNRATE"
    top = C.browse("fred")
    assert "category_id=0&" in net[-1] and top.series == ()  # the top: categories only
    assert [c.title for c in top.categories][:2] == ["Money, Banking, & Finance", "Population, Employment, & Labor Markets"]
    rates = C.browse("fred", "32447")
    assert rates.categories == () and [s.code for s in rates.series] == ["UNRATE", "UNRATENSA"]
    with pytest.raises(SpecificationError, match="a FRED category is a number"):
        C.browse("fred", "labour")


def test_world_bank_search_filters_a_sources_indicators(net):
    found = C.search("worldbank", "gdp per capita")
    assert [s.code for s in found] == ["NY.GDP.PCAP.CD", "NY.GDP.PCAP.KD.ZG"]
    assert sum("source/2/indicator" in u for u in net) == 2  # both pages of the list
    g = found[0]
    assert g.title == "GDP per capita (current US$)" and g.source == "World Development Indicators"
    assert g.last_updated == "2026-10-08"  # the source's last update (the /sources list)
    assert "Topics: Economy & Growth" in g.notes and "Source: World Bank national accounts data" in g.notes
    gdp = C.search("worldbank", "gdp")
    assert [s.code for s in gdp] == ["NY.GDP.MKTP.CD", "NY.GDP.PCAP.CD", "NY.GDP.PCAP.KD.ZG", "BX.TRF.PWKR.DT.GD.ZS"]
    assert [s.code for s in C.search("worldbank", "purchaser")] == ["NY.GDP.MKTP.CD"]  # only in the notes: still found
    assert [s.code for s in C.search("worldbank", "gdp", limit=2)] == ["NY.GDP.MKTP.CD", "NY.GDP.PCAP.CD"]
    labour = C.search("worldbank", "labor force")
    assert labour[0].code == "SL.UEM.TOTL.ZS"  # in the title first, before notes-only matches
    assert [c.path for c in C.browse("worldbank").categories] == ["sources", "topics", "countries"]
    assert [c.path for c in C.browse("worldbank", "sources").categories] == ["source/2", "source/11", "source/20"]
    assert [c.title for c in C.browse("worldbank", "topics").categories][1] == "Economy & Growth"
    assert len(C.browse("worldbank", "topic/3").series) == 5
    countries = C.browse("worldbank", "countries").categories
    assert [c.path for c in countries] == ["country/FJI", "country/TON", "country/WSM", "country/WLD"]
    assert all(c.leaf for c in countries) and countries[-1].notes == "an aggregate" and "Lower middle income" in countries[2].notes
    with pytest.raises(SpecificationError, match="The provided parameter value is not valid"):
        C.search("worldbank", "gdp", source="99")
    with pytest.raises(NotFoundError, match="nothing at 'regions'"):
        C.browse("worldbank", "regions")


def test_sdmx_dataflows_structures_and_keys(net):
    flows = C.search("sdmx", "exchange", source="ECB")
    assert [(s.code, s.title, s.kind) for s in flows] == [("ECB/EXR", "Exchange Rates", "dataflow")]  # English names
    assert flows[0].address == "sdmx://ECB/EXR/all"
    assert [s.code for s in C.search("sdmx", "financial market", source="ECB")] == ["ECB/FM", "ECB/YC"]
    assert [s.code for s in C.search("sdmx", "consumer", source="IMF")] == ["IMF/IMF.STA,CPI"]  # another agency's flow: AGENCY,ID
    with pytest.raises(SpecificationError, match="choose the SDMX provider"):
        C.search("sdmx", "exchange")
    with pytest.raises(NotFoundError, match="unknown SDMX provider 'XX'"):
        C.search("sdmx", "exchange", source="XX")
    assert [c.path for c in C.browse("sdmx").categories] == ["ECB", "ESTAT", "OECD", "IMF", "BIS"]
    assert C.browse("sdmx", "ECB").categories[1].path == "ECB/EXR" and C.browse("sdmx", "ECB").categories[1].leaf
    s = C.structure("sdmx", "ECB/EXR")
    assert net[-1].endswith("/dataflow/ECB/EXR/latest?references=all")
    assert [d.id for d in s.dimensions] == ["FREQ", "CURRENCY", "CURRENCY_DENOM", "EXR_TYPE", "EXR_SUFFIX"]  # by position
    assert s.dimension("CURRENCY").name == "Currency" and ("USD", "US dollar") in s.dimension("CURRENCY").codes
    assert s.dimension("CURRENCY_DENOM").codes == s.dimension("CURRENCY").codes  # the same codelist
    assert s.title == "Exchange Rates" and s.notes.startswith("ECB reference exchange rates")
    # the key the readers' tests read (sdmx://ECB/EXR/M.USD+GBP.EUR.SP00.A), written from the dimensions
    assert s.key(FREQ="M", CURRENCY=["USD", "GBP"], CURRENCY_DENOM="EUR", EXR_TYPE="SP00", EXR_SUFFIX="A") == "M.USD+GBP.EUR.SP00.A"
    assert s.key({"FREQ": "Q"}) == "Q...." and s.key() == "...."  # nothing chosen: any code
    assert s.address(FREQ="M", CURRENCY="USD") == "sdmx://ECB/EXR/M.USD..."
    with pytest.raises(SpecificationError, match="CURRENCY has no code 'XYZ'"):
        s.key(CURRENCY="XYZ")
    with pytest.raises(NotFoundError, match="no dimension 'REGION'"):
        s.key(REGION="X")
    with pytest.raises(SpecificationError, match="No Results Found"):
        C.structure("sdmx", "ECB/NOPE")
    with pytest.raises(SpecificationError, match="give PROVIDER/FLOW"):
        C.structure("sdmx", "EXR")
    with pytest.raises(NotFoundError, match="has no data structures"):
        C.structure("fred", "UNRATE")


def test_an_institutions_sdmx_service_is_searchable_with_its_rest_root(net):
    from pymetrix.ext.connector_search import SDMX_STRUCTURES

    try:
        ext.register_sdmx_provider("MYCB", "https://data.mycb.org/rest/data/{flow}/{key}?format=csvdata")
        with pytest.raises(SpecificationError, match="register it with structure_url"):
            C.search("sdmx", "exchange", source="MYCB")
        ext.register_sdmx_provider("MYCB", "https://data.mycb.org/rest/data/{flow}/{key}?format=csvdata",
                                   structure_url="https://data.mycb.org/rest/", agency="MYCB")
        found = C.search("sdmx", "exchange", source="mycb")
        assert net[-1] == "https://data.mycb.org/rest/dataflow/MYCB" and found[0].code == "MYCB/ECB,EXR"  # owned by ECB, not MYCB
    finally:
        ext._PROVIDERS.pop("MYCB", None)
        SDMX_STRUCTURES.pop("MYCB", None)


# -- the cache -----------------------------------------------------------------------------------------------------------


def test_the_cache_keeps_answers_for_their_time_to_live(net, monkeypatch):
    clock = {"now": 1_000_000.0}
    monkeypatch.setattr(C.time, "time", lambda: clock["now"])
    C.search("worldbank", "gdp")
    first = len(net)
    assert first == 3  # the sources list and two pages of indicators
    C.search("worldbank", "unemployment")  # another search of the same list: nothing asked
    assert len(net) == first
    assert C.cache_dir() == Path(os.environ["PYMETRIX_HOME"]) / "cache" / "connectors" and len(C.cache_entries("worldbank")) == 3
    clock["now"] += 6 * 24 * 3600  # six days: metadata keep for a week
    C.search("worldbank", "gdp")
    assert len(net) == first
    clock["now"] += 2 * 24 * 3600  # eight days: asked again
    C.search("worldbank", "gdp")
    assert len(net) == 2 * first
    C.search("worldbank", "gdp", refresh=True)  # refresh: asked again however fresh
    assert len(net) == 3 * first
    C.set_api_key("fred", KEY)
    C.search("fred", "unemployment")
    C.search("fred", "unemployment")
    assert sum("series/search" in u for u in net) == 1
    clock["now"] += 23 * 3600
    C.search("fred", "unemployment")
    assert sum("series/search" in u for u in net) == 1
    clock["now"] += 2 * 3600  # a search keeps for a day
    C.search("fred", "unemployment")
    assert sum("series/search" in u for u in net) == 2
    assert {e["kind"] for e in C.cache_entries("fred")} == {"search"}
    settings = C.configure_cache(search_ttl=60)
    try:
        assert settings["search_ttl"] == 60 and settings["folder"] == str(C.cache_dir())
        clock["now"] += 61
        C.search("fred", "unemployment")
        assert sum("series/search" in u for u in net) == 3
        C.configure_cache(enabled=False)
        assert C.clear_cache() == 4 and C.cache_entries() == []
        C.search("fred", "unemployment")
        assert C.cache_entries() == [] and sum("series/search" in u for u in net) == 4  # off: nothing kept
    finally:
        C.configure_cache(search_ttl=24 * 3600, enabled=True)
    Path(C.cache_dir() / "fred-broken.json").write_text("{not json", encoding="utf-8")  # a damaged entry is asked again
    C.search("fred", "unemployment")
    assert sum("series/search" in u for u in net) == 5


def test_no_key_is_ever_written_to_disk_or_shown(net, monkeypatch):
    C.set_api_key("fred", KEY)
    C.search("fred", "unemployment")
    C.browse("fred", "32447")
    home = Path(os.environ["PYMETRIX_HOME"])
    files = [p for p in home.rglob("*") if p.is_file()]
    assert len(files) == 3 and not any(KEY in p.read_text(encoding="utf-8") or KEY in p.name for p in files)
    assert all("api_key" not in e["request"] for e in C.cache_entries())
    assert C.redact(f"https://x.org/a?b=1&api_key={KEY}&token=t") == "https://x.org/a?b=1"

    def refused(url):  # the readers' transport puts the URL in its errors
        raise ConnectionError(f"HTTP 400 Bad Request ({url})")

    monkeypatch.setattr(ext, "_get", refused)
    with pytest.raises(ConnectionError) as err:
        C.search("fred", "inflation")
    assert KEY not in str(err.value) and "api_key=***" in str(err.value)

    def echoing(url):  # an answer that repeats the key is kept without it
        return (SEARCH / "fred_bad_key.json").read_text().replace("variable api_key", f"variable api_key {KEY}").encode()

    monkeypatch.setattr(ext, "_get", echoing)
    with pytest.raises(SpecificationError, match="FRED: Bad Request") as err:
        C.search("fred", "prices")
    assert not any(KEY in p.read_text(encoding="utf-8") for p in home.rglob("*.json"))
    C.set_api_key("fred", None)
    assert C.api_key("fred") == ""


# -- the browser ---------------------------------------------------------------------------------------------------------


@pytest.fixture()
def qapp():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _same(a, b):
    assert a.names == b.names and a.frame.equals(b.frame)


def _replay_without_the_browser(s, name, monkeypatch):
    """The recorded script, run with no searchers, no cache and the browser's module unimportable."""
    script = s.script()
    assert "connectors." not in script and "browse" not in script.lower()
    monkeypatch.setattr(C, "_SEARCHERS", {})
    monkeypatch.setitem(sys.modules, "pymetrix.ui.series_browser_qt", None)
    C.clear_cache()
    ns: dict = {}
    exec(script, ns)
    return ns[name]


def test_the_browser_fred_search_select_load(net, qapp, monkeypatch):
    from pymetrix.ui.series_browser_qt import series_browser

    s = Session()
    loaded, edited = [], []
    b = series_browser(s, on_load=loaded.append, on_edit=edited.append)
    b.set_provider("fred")
    assert b.key.isVisibleTo(b.dialog) and not b.key_box.isVisibleTo(b.dialog) and "API key" in b.note.text()
    assert b.search("unemployment") == [] and "needs an API key" in b.last_error  # said in the dialog, not raised
    b.key.setText(KEY)  # for this session only
    found = b.search("unemployment")
    assert b.table.rowCount() == 4 and b.table.item(0, 1).text() == "UNRATE" and b.table.item(0, 2).text() == "Monthly"
    assert [b.table.horizontalHeaderItem(i).text() for i in range(4)] == ["Title", "Code", "Frequency", "Units"]
    b.select_rows([0, 2])  # monthly and quarterly
    with pytest.raises(SpecificationError, match="Monthly and Quarterly"):
        b.command()
    b.select_rows([0])
    assert "labor force" in b.details.toPlainText() and "fred://UNRATE" in b.details.toPlainText()
    b.name.setText("u")
    cmd = b.add_to_load()
    assert edited == [cmd] and cmd == LoadFromConnector("fred", "UNRATE", name="u")  # fills Load from connector…
    assert b.load() == cmd and loaded == ["u"]
    _same(s["u"], Dataset.read(found[0].address))  # what the browser loads is what the address reads
    assert "u = Dataset.read('fred://UNRATE', vintage='latest')" in s.script()
    assert KEY not in s.script()
    _same(_replay_without_the_browser(s, "u", monkeypatch), s["u"])


def test_the_browser_world_bank_with_countries(net, qapp, monkeypatch):
    from pymetrix.ui.series_browser_qt import series_browser

    s = Session()
    b = series_browser(s)
    b.set_provider("worldbank")
    assert b.country_box.isVisibleTo(b.dialog) and b.countries.count() == 4 and b.source_value() == "2"
    assert [b.tree.topLevelItem(i).text(0) for i in range(3)] == ["Sources", "Topics", "Countries"]
    b._open_item(b.tree.topLevelItem(1))  # Topics: its topics under it
    topic = b.tree.topLevelItem(1).child(1)
    b._open_item(topic)  # a topic: its indicators in the table
    assert topic.text(0) == "Economy & Growth" and b.table.rowCount() == 5
    b.search("gdp current")
    b.select_rows([0])
    b.choose_countries(["WSM", "TON"])
    b.name.setText("wb")
    cmd = b.command()
    assert cmd == LoadFromConnector("worldbank", "NY.GDP.MKTP.CD", name="wb", options="country=TON;WSM")
    b.load()
    _same(s["wb"], Dataset.read("worldbank://NY.GDP.MKTP.CD?country=TON;WSM"))
    assert s["wb"].kind == "panel"
    _same(_replay_without_the_browser(s, "wb", monkeypatch), s["wb"])


def test_the_browser_sdmx_dataflow_dimensions_key(net, qapp, monkeypatch):
    from pymetrix.ui.series_browser_qt import series_browser

    s = Session()
    b = series_browser(s)
    b.set_provider("sdmx")
    assert b.key_box.isVisibleTo(b.dialog) and not b.key.isVisibleTo(b.dialog) and b.source_value() == "ECB"
    assert b.tree.topLevelItemCount() == 6  # the ECB's dataflows
    with pytest.raises(SpecificationError, match="choose a dataflow"):
        b.command()
    b.search("exchange rates")
    b.select_rows([0])  # a dataflow: its dimensions become pickers
    assert list(b.pickers) == ["FREQ", "CURRENCY", "CURRENCY_DENOM", "EXR_TYPE", "EXR_SUFFIX"]
    assert b.key_preview.text() == "sdmx://ECB/EXR/...."
    b.choose("FREQ", ["M"])
    b.choose("CURRENCY", ["USD", "GBP"])
    b.choose("CURRENCY_DENOM", ["EUR"])
    b.choose("EXR_TYPE", ["SP00"])
    b.choose("EXR_SUFFIX", ["A"])
    assert b.key_preview.text() == "sdmx://ECB/EXR/M.GBP+USD.EUR.SP00.A"  # codes in the codelist's order
    with pytest.raises(SpecificationError, match="no dimension 'REGION'"):
        b.choose("REGION", ["X"])
    b.name.setText("fx")
    assert b.command() == LoadFromConnector("sdmx", "ECB/EXR/M.GBP+USD.EUR.SP00.A", name="fx")
    b.load()
    direct = Dataset.read("sdmx://ECB/EXR/M.GBP+USD.EUR.SP00.A")
    _same(s["fx"], direct)
    assert "fx = Dataset.read('sdmx://ECB/EXR/M.GBP+USD.EUR.SP00.A', vintage='latest')" in s.script()
    _same(_replay_without_the_browser(s, "fx", monkeypatch), direct)


def test_file_menu_opens_the_browser_and_add_fills_load_from_connector(net, qapp, monkeypatch):
    from pymetrix.commands import COMMANDS
    from pymetrix.commands.menus import menu_entries
    from pymetrix.help import topic
    from pymetrix.ui import qt as ui

    entries = menu_entries("File", [i for i in COMMANDS.values() if i.menu])
    labels = [e.label if e.kind != "action" else "@" + e.action for e in entries]
    assert labels[labels.index("Load from connector…") + 1] == "@series-browser"
    assert "File → Browse series" in topic("data-connectors").text
    opened = []
    monkeypatch.setattr(ui, "command_dialog", lambda cls, session, parent=None, initial=None: opened.append(initial) or
                        type("D", (), {"exec": lambda self: 0})())
    window = ui.main_window(Session())
    try:
        file_menu = window.command_menus["File"]
        action = next(a for a in file_menu.actions() if a.text().replace("&", "") == "Browse series…")
        action.trigger()
        b = window.series_browser
        b.set_provider("sdmx")
        b.show_structure("ECB/EXR")
        b.choose("FREQ", ["M"])
        b.edit_button.click()
        assert opened == [LoadFromConnector("sdmx", "ECB/EXR/M....", name="data")]
    finally:
        window.close()
