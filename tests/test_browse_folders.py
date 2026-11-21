"""Folders in File → Browse series… (pymetrix.ext.browse_folders): an SDMX provider's dataflows in its own category tree,
the rule that falls back to name families, ID prefixes or a flat list, World Bank indicators grouped by code prefix,
and FRED's search order with sortable columns.

Nothing here uses the network, and none of it has been checked against the providers' live answers (they could not be
reached when it was written). The category schemes are written by hand from the SDMX 2.1 standard, each fixture's
comment saying what it holds: ecb_categoryscheme.xml (ECB-like: nested categories referred to by path, by a nested id
and by URN; a dataflow in two categories; one in none; categorisations to ignore), estat_categoryscheme.xml with
estat_dataflows.xml (Eurostat-like: two schemes, three levels, upper-case IDs with theme prefixes) and
imf_categoryscheme_coarse.xml (one category holding 44 of the IMF's 46 dataflows). wb_topic1_indicators.json is a World
Bank topic list of 45 indicators with the World Development Indicators' codes; fred_series_search_gdp.json a FRED search
answer in FRED's documented JSON (order_by search_rank, sort_order desc). The expected trees are written out by hand
from the fixtures, not computed by the module.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_series_browser import KEY, _replay_without_the_browser, _same, net, qapp  # noqa: F401 - requests answered from fixtures

from pymetrix import Dataset
from pymetrix import connectors as C
from pymetrix.commands import LoadFromConnector, Session
from pymetrix.ext import connectors as ext
from pymetrix.ext.browse_folders import (Categories, DataflowTree, Folder, choose_tree, code_prefix_groups, flow_id, flow_key,
                                         id_prefix, judge, name_tree, parse_categories, prefix_tree)
from pymetrix.ext.connector_search import SdmxSearch, dataflow_tree
from pymetrix.ext.dataflow_groups import GroupedDataflow, group_dataflows

SEARCH = Path(__file__).parent / "fixtures" / "connectors" / "search"
ECB_CATS = SEARCH / "ecb_categoryscheme.xml"
ESTAT_CATS = SEARCH / "estat_categoryscheme.xml"
IMF_COARSE = SEARCH / "imf_categoryscheme_coarse.xml"
ECB_ROOT = "https://data-api.ecb.europa.eu/service"
ESTAT_ROOT = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1"


def _flows(provider: str, fixture: str) -> list[tuple[str, str]]:
    return [(s.code, s.title) for s in SdmxSearch().dataflows(provider, lambda url, kind: (SEARCH / fixture).read_bytes())]


def ecb() -> list[tuple[str, str]]:
    return _flows("ECB", "ecb_dataflows.xml")


def estat() -> list[tuple[str, str]]:
    return _flows("ESTAT", "estat_dataflows.xml")


def imf() -> list[tuple[str, str]]:
    return _flows("IMF", "imf_dataflows_long.xml")


def shape(node: Folder) -> list:
    """A folder's children as [title, [...]] for folders and the code for dataflows."""
    return [[c.title, shape(c)] if isinstance(c, Folder) else c.code for c in node.children]


def codes_in(node: Folder) -> list[str]:
    return [x for c in node.children for x in (codes_in(c) if isinstance(c, Folder) else [c.code])]


# -- the provider's categories --------------------------------------------------------------------------------------------


def test_an_ecb_like_category_scheme_puts_dataflows_in_nested_folders():
    cats = parse_categories(ECB_CATS.read_bytes())
    assert list(cats.schemes) == [("ECB", "MOBILE_NAVI")] and cats.schemes[("ECB", "MOBILE_NAVI")][0] == "Economic concepts"
    assert len(cats.links) == 8  # the data structure's categorisation is not a dataflow's: left out when read
    tree = choose_tree(ecb(), cats, "ECB")
    assert (tree.grouping, tree.why) == ("categories", "")
    assert shape(tree.root) == [
        ["Prices, output, demand and labour market", [["Prices", ["ECB/ICP"]]]],  # by path, 05.05_01
        ["Monetary and financial statistics", [["Balance sheet items", ["ECB/BSI"]],  # by path
                                               ["MFI interest rates", ["ECB/MIR"]]]],  # by its own id, 06_02
        ["Financial markets and interest rates", [["Yield curves", ["ECB/YC"]], "ECB/EXR"]],  # by URN; sub-folders first
        ["Exchange rates", ["ECB/EXR"]],  # EXR is in two categories: in both
        ["Other", ["ECB/FM"]],  # its only categorisation names a category that does not exist
    ]  # "Monetary operations" and "Banking supervision" hold no listed dataflow (RETIRED is not listed): left out
    prices = tree.root.folders[0]
    assert prices.folders[0].notes == "consumer and producer prices" and prices.folders[0].flows[0].label == "Indices of Consumer prices"
    assert sorted(set(codes_in(tree.root))) == sorted(code for code, _ in ecb())  # every dataflow listed is in the tree


def test_a_eurostat_like_scheme_three_deep_with_a_second_scheme_and_other():
    flows = estat()
    assert len(flows) == 36 and flows[0] == ("ESTAT/NAMA_10_GDP", "GDP and main components (output, expenditure and income)")
    tree = choose_tree(flows, parse_categories(ESTAT_CATS.read_bytes()), "ESTAT")
    assert tree.grouping == "categories" and tree.note == "Grouped by the provider's categories"
    top = [f.title for f in tree.root.folders]
    # the main scheme's categories (its order, not sorted), then the other scheme as one folder, then Other
    assert top == ["Economy and finance", "Population and social conditions", "Industry, trade and services",
                   "Sustainable development indicators", "Other"]
    economy = tree.root.folders[0]
    assert [f.title for f in economy.folders] == ["National accounts (ESA 2010)", "Prices", "Government statistics",
                                                  "Balance of payments", "Euro indicators"]
    annual = economy.folders[0].folders[0]
    assert annual.title == "Annual national accounts" and [r.code for r in annual.flows] == [
        "ESTAT/NAMA_10_CO3_P3", "ESTAT/NAMA_10_GDP", "ESTAT/NAMA_10_A10", "ESTAT/NAMA_10_PC"]  # by name, ignoring case
    assert shape(tree.root.folders[3]) == [["Decent work and economic growth", ["ESTAT/NAMA_10_PC", "ESTAT/SDG_08_10"]]]
    assert shape(tree.root.folders[4]) == ["ESTAT/TEC00001", "ESTAT/TEINA010", "ESTAT/TIPSUN20"]
    assert "Agriculture, forestry and fisheries" not in str(shape(tree.root))  # an empty category is left out
    assert len(economy.codes()) == 23 and len(tree.root.codes()) == 36
    placed = codes_in(tree.root)
    assert sorted(set(placed)) == sorted(c for c, _ in flows) and placed.count("ESTAT/NAMA_10_PC") == 2


# -- the rule: categories, else names, else ID prefixes, else flat -------------------------------------------------------


def _tree(sizes: list[int], loose: int = 0, grouping: str = "names") -> DataflowTree:
    """Folders of the given sizes and ``loose`` rows of their own."""
    n = iter(range(10_000))
    folders = [Folder(f"F{i}", tuple(GroupedDataflow(f"X/{next(n)}", "x", "x") for _ in range(k))) for i, k in enumerate(sizes)]
    rows = [GroupedDataflow(f"X/{next(n)}", "x", "x") for _ in range(loose)]
    return DataflowTree(Folder("", (*folders, *rows)), grouping)


def test_the_choice_rule_at_its_edges():
    assert judge(_tree([10]), 10) == "fewer than two folders"
    assert judge(_tree([5, 5]), 10) == ""
    assert judge(_tree([3, 2], loose=5), 10) == ""  # exactly half in folders: enough
    assert judge(_tree([2, 2], loose=5), 9) == "its folders hold fewer than half the dataflows"  # 4 of 9
    assert judge(_tree([8, 2]), 10) == ""  # exactly four fifths in one folder: allowed
    assert judge(_tree([9, 1]), 10) == "one folder holds more than four fifths of the dataflows"
    # for categories, "Other" is not a folder of the provider's
    other = DataflowTree(Folder("", (*_tree([3, 1]).root.folders, Folder("Other", (GroupedDataflow("X/o", "o", "o"),)))), "categories")
    assert judge(other, 5) == ""
    lone = DataflowTree(Folder("", (_tree([3]).root.folders[0], Folder("Other", tuple(GroupedDataflow(f"X/o{i}", "o", "o") for i in range(3))))),
                        "categories")
    assert judge(lone, 6) == "fewer than two folders"


def test_a_coarse_imf_scheme_gives_way_to_the_name_families_as_before():
    flows = imf()
    coarse = choose_tree(flows, parse_categories(IMF_COARSE.read_bytes()))
    # "All datasets" is opened (it holds only folders); then Statistics holds 44 of 46: too coarse
    assert coarse.grouping == "names" and "one folder holds more than four fifths" in coarse.why
    step1 = [[f.title, [x.code for x in f.flows]] if f.folder else f.flows[0].code for f in group_dataflows(flows)]
    assert shape(coarse.root) == step1  # the tree step 1 showed, unchanged
    missing = choose_tree(flows, None, why="No Results Found")
    assert missing.grouping == "names" and shape(missing.root) == step1 and missing.why == "no categories (No Results Found)"
    empty = choose_tree(flows, Categories())
    assert empty.grouping == "names" and empty.why == "the provider's categories hold no dataflows"


def test_without_categories_eurostat_by_id_prefix_and_the_ecb_flat():
    tree = choose_tree(estat(), None, "ESTAT")
    assert tree.grouping == "prefixes" and "by name: fewer than two folders" in tree.why
    assert [f.title for f in tree.root.folders] == [
        "bop_ Balance of payments", "demo_ Population", "ei_ Euro indicators", "gov_ Government finance",
        "nama_ National accounts", "namq_ National accounts, quarterly", "prc_ Prices", "sts_ Short-term business statistics",
        "une_ Unemployment"]
    assert [r.code for r in tree.root.folders[4].flows] == ["ESTAT/NAMA_10_CO3_P3", "ESTAT/NAMA_10_GDP", "ESTAT/NAMA_10_A10",
                                                            "ESTAT/NAMA_10_PC"]
    # a prefix of one dataflow (NASA_, LFSI_, SDG_) and an ID with no underscore are rows, after the folders, by name
    assert [r.code for r in tree.root.flows] == ["ESTAT/LFSI_EMP_A", "ESTAT/LFSQ_EGAN", "ESTAT/TEC00001", "ESTAT/TEINA010",
                                                 "ESTAT/NASA_10_NF_TR", "ESTAT/NASQ_10_NF_TR", "ESTAT/SDG_08_10", "ESTAT/TIPSUN20"]
    unknown = prefix_tree([("P/XYZ_1", "b"), ("P/XYZ_2", "a"), ("P/NAMA_1", "c")])
    assert [f.title for f in unknown.root.folders] == ["xyz_"] and unknown.root.folders[0].notes == "dataflows whose IDs start XYZ_"
    flat = choose_tree(ecb(), None, "ECB", why="HTTP 403 Forbidden")
    assert flat.grouping == "flat" and flat.root.folders == []
    assert [r.label for r in flat.root.flows] == sorted((t for _, t in ecb()), key=str.casefold)
    # neither useful, but the names make a folder: kept, as step 1 did
    few = [("P/A", "Survey (S), one"), ("P/B", "Survey (S), two"), ("P/C", "c"), ("P/D", "d"), ("P/E", "e")]
    some = choose_tree(few, None)
    assert some.grouping == "names" and shape(some.root) == ["P/C", "P/D", "P/E", ["Survey (S)", ["P/A", "P/B"]]]


def test_ids_and_prefixes():
    assert flow_id("ECB/EXR") == "EXR" and flow_id("IMF/IMF.STA,CPI") == "CPI" and flow_id("IMF/IMF.STA,CPI,1.0") == "CPI"
    assert flow_key("ECB/EXR", "ECB") == ("ECB", "EXR") and flow_key("IMF/IMF.STA,CPI") == ("IMF.STA", "CPI")
    assert id_prefix("NAMA_10_GDP") == "nama" and id_prefix("nama_10_gdp") == "nama" and id_prefix("EXR") == ""
    assert id_prefix("_X") == "" and name_tree([]).root.children == () and choose_tree([], None).grouping == "flat"


# -- asking the provider ----------------------------------------------------------------------------------------------------


@pytest.fixture()
def serve(net, monkeypatch):  # noqa: F811
    """Answers for the category requests (and Eurostat's dataflows), before test_series_browser's routes."""
    routes: dict[str, object] = {f"{ESTAT_ROOT}/dataflow/ESTAT": SEARCH / "estat_dataflows.xml"}
    others = ext._get

    def fake(url: str) -> bytes:
        for pattern, answer in routes.items():
            if pattern in url:
                net.append(url)
                if isinstance(answer, Exception):
                    raise answer
                return answer if isinstance(answer, bytes) else Path(answer).read_bytes()
        return others(url)

    monkeypatch.setattr(ext, "_get", fake)
    return routes, net


def test_the_category_request_is_kept_a_week_and_a_failure_falls_back(serve):
    routes, asked = serve
    url = f"{ECB_ROOT}/categoryscheme/ECB/all/latest?references=parentsandsiblings"
    routes[url] = ECB_CATS
    tree = dataflow_tree("ECB")
    assert tree.grouping == "categories" and url in asked and f"{ECB_ROOT}/dataflow/ECB" in asked
    kept = {e["request"]: e["kind"] for e in C.cache_entries("sdmx")}
    assert kept[url] == "metadata"  # kept a week, as other metadata
    before = len(asked)
    assert shape(dataflow_tree("ECB", ecb()).root) == shape(tree.root) and len(asked) == before  # from the cache
    dataflow_tree("ECB", ecb(), refresh=True)
    assert asked[before:] == [url]
    for failure, said in ((ConnectionError("HTTP 403 Forbidden (proxy)"), "HTTP 403 Forbidden"),
                          (b"Service unavailable", "no categories ("),
                          ((SEARCH / "sdmx_no_flow.xml").read_bytes(), "No Results Found")):
        routes[url] = failure
        fallen = dataflow_tree("ECB", ecb(), refresh=True)
        assert fallen.grouping == "flat" and said in fallen.why, fallen.why
    routes[f"{ESTAT_ROOT}/categoryscheme/ESTAT/all/latest?references=parentsandsiblings"] = ESTAT_CATS
    assert dataflow_tree("ESTAT").grouping == "categories"
    routes["api.imf.org/external/sdmx/2.1/categoryscheme/all/all/latest?references=parentsandsiblings"] = IMF_COARSE
    routes["api.imf.org/external/sdmx/2.1/dataflow/all"] = SEARCH / "imf_dataflows_long.xml"
    assert dataflow_tree("IMF").grouping == "names"


# -- World Bank: long lists by code prefix -------------------------------------------------------------------------------------


@pytest.fixture()
def wb(serve):
    routes, asked = serve
    routes["worldbank.org/v2/topic/1/indicator"] = SEARCH / "wb_topic1_indicators.json"
    return asked


def test_a_long_world_bank_topic_is_grouped_by_code_prefix(wb):
    everything = json.loads((SEARCH / "wb_topic1_indicators.json").read_text())[1]
    listing = C.browse("worldbank", "topic/1")
    assert len(listing.series) == len(everything) == 45  # the whole list still, in the provider's order
    assert [s.code for s in listing.series] == [r["id"] for r in everything]
    assert [(c.path, c.title) for c in listing.categories] == [
        ("topic/1/AG", "AG. Agriculture and land"), ("topic/1/AGRIC", "AGRIC."), ("topic/1/DC", "DC."),
        ("topic/1/EG", "EG. Energy"), ("topic/1/EN", "EN. Environment: emissions"), ("topic/1/ER", "ER. Environment: land and water"),
        ("topic/1/NV", "NV. National accounts: value added"), ("topic/1/SH", "SH. Health"),
        ("topic/1/SI", "SI. Poverty and income distribution"), ("topic/1/SL", "SL. Labour"), ("topic/1/SP", "SP. Population"),
        ("topic/1/TM", "TM. Trade: imports"), ("topic/1/TX", "TX. Trade: exports")]
    assert all(c.leaf for c in listing.categories) and listing.categories[0].notes == "20 indicators whose codes start AG."
    ag = C.browse("worldbank", "topic/1/AG").series
    assert [s.code for s in ag] == [r["id"] for r in everything if r["id"].startswith("AG.")] and len(ag) == 20
    assert C.browse("worldbank", "topic/1/AGRIC").series[0].code == "AGRIC.INV.SHR"  # AGRIC. is not AG.
    parts = [len(C.browse("worldbank", c.path).series) for c in listing.categories]
    assert sum(parts) == 45  # every indicator in one folder
    assert C.browse("worldbank", "topic/3").categories == ()  # a short list (5) is not grouped
    assert C.browse("worldbank", "source/2").categories == ()  # nor is this short source list (6)
    assert code_prefix_groups(["NY.GDP", "SL.UEM", "NY.GNP", "XX"], {"NY": "National accounts"}) == [
        ("NY", "NY. National accounts", ["NY.GDP", "NY.GNP"]), ("SL", "SL.", ["SL.UEM"]), ("XX", "XX.", ["XX"])]


# -- FRED: relevance, then sortable columns -------------------------------------------------------------------------------------


def test_fred_search_asks_for_and_keeps_freds_relevance_order(serve, monkeypatch):
    routes, asked = serve
    routes["fred/series/search?search_text=gdp&"] = SEARCH / "fred_series_search_gdp.json"
    monkeypatch.setenv("FRED_API_KEY", KEY)
    found = C.search("fred", "gdp")
    assert "order_by=search_rank" in asked[-1] and "sort_order=desc" in asked[-1]
    answer = json.loads((SEARCH / "fred_series_search_gdp.json").read_text())
    assert (answer["order_by"], answer["sort_order"]) == ("search_rank", "desc")  # FRED says how it ordered them
    assert [s.code for s in found] == [s["id"] for s in answer["seriess"]]  # and that order is kept


# -- the browser --------------------------------------------------------------------------------------------------------------


@pytest.fixture()
def browser(serve, qapp):
    from pymetrix.ui.series_browser_qt import series_browser

    routes, asked = serve
    s = Session()
    b = series_browser(s)
    yield s, b, routes, asked
    b.dialog.close()


def _sdmx(b, provider: str) -> None:
    b.set_provider("sdmx")
    b.source.setCurrentIndex(b.source.findText(provider))


def _child(item, title):
    return next(item.child(j) for j in range(item.childCount()) if item.child(j).text(0) == title)


def _top(b, title):
    return next(b.tree.topLevelItem(i) for i in range(b.tree.topLevelItemCount()) if b.tree.topLevelItem(i).text(0) == title)


def _tree_shape(item) -> list:
    from PySide6.QtCore import Qt

    kids = [item.child(j) for j in range(item.childCount())]
    return [[k.text(0), _tree_shape(k)] if k.childCount() else k.data(0, Qt.ItemDataRole.UserRole) for k in kids]


def test_the_browser_shows_the_ecb_categories_and_says_so(browser):
    _s, b, routes, _asked = browser
    routes[f"{ECB_ROOT}/categoryscheme/ECB/all/latest?references=parentsandsiblings"] = ECB_CATS
    _sdmx(b, "ECB")
    assert b.grouping == "categories" and b.tree.headerItem().text(0) == "Browse · by category"
    assert b.tree.headerItem().toolTip(0) == "Grouped by the provider's categories."
    expected = choose_tree(ecb(), parse_categories(ECB_CATS.read_bytes()), "ECB")
    assert _tree_shape(b.tree.invisibleRootItem()) == shape(expected.root)
    markets = _top(b, "Financial markets and interest rates")
    assert markets.toolTip(0) == "Financial markets and interest rates\n2 dataflows"
    prices = _child(_top(b, "Prices, output, demand and labour market"), "Prices")
    assert prices.toolTip(0) == "Prices\n1 dataflow\n\nConsumer and producer prices"
    yc = _child(_child(markets, "Yield curves"), "Financial market data - yield curve")
    assert yc.toolTip(0).splitlines()[:2] == ["Financial market data - yield curve", "Dataflow YC"]
    b.set_provider("fred")
    assert b.grouping == "" and b.tree.headerItem().text(0) == "Browse"  # another provider: no note


def test_the_filter_works_across_nested_folders(browser):
    _s, b, routes, _asked = browser
    routes[f"{ESTAT_ROOT}/categoryscheme/ESTAT/all/latest?references=parentsandsiblings"] = ESTAT_CATS
    _sdmx(b, "ESTAT")
    economy = _top(b, "Economy and finance")
    prices = _child(economy, "Prices")
    hicp = _child(prices, "Harmonised index of consumer prices (HICP)")
    euro = _child(economy, "Euro indicators")
    euro.setExpanded(True)  # opened by hand: stays open
    b.query.setText("hicp")  # typed: filters as it is typed
    assert b.filter_tree() == 4  # three under Prices, one under Euro indicators, three levels down
    shown = [hicp.child(j).text(0) for j in range(hicp.childCount()) if not hicp.child(j).isHidden()]
    assert shown == ["HICP - annual data (average index and rate of change)", "HICP - monthly data (annual rate of change)",
                     "HICP - monthly data (index)"]
    assert economy.isExpanded() and prices.isExpanded() and hicp.isExpanded()  # every folder down to them is opened
    assert [euro.child(j).text(0) for j in range(euro.childCount()) if not euro.child(j).isHidden()] == ["HICP (2015 = 100) - monthly data"]
    assert _child(prices, "Purchasing power parities").isHidden() and _child(economy, "Government statistics").isHidden()
    assert _top(b, "Population and social conditions").isHidden() and _top(b, "Other").isHidden()
    assert b.filter_tree("per capita") == 3  # NAMA_10_PC, NAMQ_10_PC, SDG_08_10: NAMA_10_PC is shown twice, counted once
    rows, _folders = b._items()
    assert sorted(r.data(0, 0x0100) for r in rows if not r.isHidden()) == [
        "ESTAT/NAMA_10_PC", "ESTAT/NAMA_10_PC", "ESTAT/NAMQ_10_PC", "ESTAT/SDG_08_10"]
    assert not _top(b, "Sustainable development indicators").isHidden()
    assert b.filter_tree("nama_10_gdp") == 1  # by code
    b.query.setText("")
    assert not economy.isExpanded() and not prices.isExpanded() and not hicp.isExpanded()  # what the filter opened is closed
    assert euro.isExpanded()  # what was opened by hand is not
    assert all(not b.tree.topLevelItem(i).isHidden() for i in range(b.tree.topLevelItemCount()))


def test_loading_from_a_nested_folder_records_what_the_address_reads(browser, monkeypatch):
    s, b, routes, _asked = browser
    routes[f"{ECB_ROOT}/categoryscheme/ECB/all/latest?references=parentsandsiblings"] = ECB_CATS
    _sdmx(b, "ECB")
    exr = _child(_top(b, "Financial markets and interest rates"), "Exchange Rates")  # a row beside a sub-folder
    b.tree.itemClicked.emit(exr, 0)  # as a click does
    assert b.structure is not None and b.structure.flow == "EXR"
    for dim, codes in (("FREQ", ["M"]), ("CURRENCY", ["USD", "GBP"]), ("CURRENCY_DENOM", ["EUR"]), ("EXR_TYPE", ["SP00"]),
                       ("EXR_SUFFIX", ["A"])):
        b.choose(dim, codes)
    address = "sdmx://ECB/EXR/M.GBP+USD.EUR.SP00.A"
    assert b.key_preview.text() == address
    b.name.setText("fx")
    assert b.command() == LoadFromConnector("sdmx", address.partition("://")[2], name="fx")
    assert b.load() is not None
    direct = Dataset.read(address)
    _same(s["fx"], direct)
    assert f"fx = Dataset.read({address!r}, vintage='latest')" in s.script()
    _same(_replay_without_the_browser(s, "fx", monkeypatch), direct)


def test_the_header_says_when_the_tree_is_by_name_or_id_prefix(browser):
    _s, b, routes, _asked = browser
    routes[f"{ESTAT_ROOT}/categoryscheme/ESTAT/all/latest?references=parentsandsiblings"] = ConnectionError("HTTP 403 Forbidden")
    _sdmx(b, "ESTAT")
    assert b.grouping == "prefixes" and b.tree.headerItem().text(0) == "Browse · by ID prefix"
    assert b.tree.headerItem().toolTip(0).startswith("Grouped by the start of the dataflow's ID.\nNot by the provider's categories: "
                                                     "no categories (HTTP 403 Forbidden)")
    assert b.tree.topLevelItem(0).text(0) == "bop_ Balance of payments"
    routes["api.imf.org/external/sdmx/2.1/categoryscheme/"] = IMF_COARSE
    routes["api.imf.org/external/sdmx/2.1/dataflow/all"] = SEARCH / "imf_dataflows_long.xml"
    b.source.setCurrentIndex(b.source.findText("IMF"))
    assert b.grouping == "names" and b.tree.headerItem().text(0) == "Browse · by name"
    assert "four fifths" in b.tree.headerItem().toolTip(0)
    assert [b.tree.topLevelItem(i).text(0) for i in range(b.tree.topLevelItemCount())] == [f.title for f in group_dataflows(imf())]


def test_the_browser_world_bank_topic_has_a_folder_per_code_prefix(browser, wb):
    _s, b, _routes, _asked = browser
    b.set_provider("worldbank")
    topics = b.tree.topLevelItem(1)
    b._open_item(topics)
    topic = next(topics.child(j) for j in range(topics.childCount()) if topics.child(j).text(0).startswith("Agriculture"))
    b._open_item(topic)
    assert b.table.rowCount() == 45  # the whole topic, as before
    assert [topic.child(j).text(0) for j in range(3)] == ["AG. Agriculture and land", "AGRIC.", "DC."]
    b._open_item(topic.child(0))
    assert b.table.rowCount() == 20 and b.table.item(0, 1).text() == "AG.LND.AGRI.ZS" and "20 series" in b.status.text()
    b.select_rows([0])
    b.name.setText("land")
    assert b.command() == LoadFromConnector("worldbank", "AG.LND.AGRI.ZS", name="land")


def test_the_table_sorts_by_a_clicked_column_and_back_to_freds_order(browser, monkeypatch):
    s, b, routes, _asked = browser
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    routes["fred/series/search?search_text=gdp&"] = SEARCH / "fred_series_search_gdp.json"
    b.set_provider("fred")
    b.key.setText(KEY)
    b.search("gdp")
    relevance = ["GDP", "GDPC1", "A191RL1Q225SBEA", "GDPPOT", "GDPDEF", "NYGDPMKTPCDWLD"]  # FRED's order
    codes = lambda: [b.table.item(r, 1).text() for r in range(b.table.rowCount())]  # noqa: E731
    assert codes() == relevance and not b.table.horizontalHeader().isSortIndicatorShown()
    b.select_rows([1])  # GDPC1
    header = b.table.horizontalHeader()
    at = QPoint(header.sectionViewportPosition(0) + header.sectionSize(0) // 2, header.height() // 2)
    QTest.mouseClick(header.viewport(), Qt.MouseButton.LeftButton, pos=at)  # a click on Title's heading
    by_title = ["GDP", "NYGDPMKTPCDWLD", "GDPDEF", "GDPC1", "A191RL1Q225SBEA", "GDPPOT"]
    assert codes() == by_title  # ties ("Real Gross Domestic Product") keep FRED's order
    assert by_title == [x.code for x in sorted(C.search("fred", "gdp"), key=lambda x: x.title.casefold())]  # Python's stable sort agrees
    assert header.isSortIndicatorShown() and header.sortIndicatorSection() == 0
    assert header.sortIndicatorOrder() == Qt.SortOrder.AscendingOrder
    assert [x.code for x in b.selected()] == ["GDPC1"]  # the selection follows the row
    b.sort_by(0)
    assert codes() == ["GDPPOT", "GDPC1", "A191RL1Q225SBEA", "GDPDEF", "NYGDPMKTPCDWLD", "GDP"]
    assert header.sortIndicatorOrder() == Qt.SortOrder.DescendingOrder
    b.sort_by(0)
    assert codes() == relevance and not header.isSortIndicatorShown()  # a third click: FRED's order again
    b.sort_by(2)  # Frequency: Annual before Quarterly
    assert codes()[0] == "NYGDPMKTPCDWLD" and codes()[1:] == relevance[:-1]
    b.sort_by(6)  # Updated, as dates written YYYY-MM-DD
    assert codes()[:2] == ["GDPPOT", "NYGDPMKTPCDWLD"]
    assert b.table.horizontalHeaderItem(0).toolTip().startswith("Click to sort by this column")
    assert [x.code for x in b.selected()] == ["GDPC1"]
    b.name.setText("y")
    assert b.command() == LoadFromConnector("fred", "GDPC1", name="y")  # the sorted row's series, not the row number's
    assert b.load() is not None
    _same(s["y"], Dataset.read("fred://GDPC1"))
    assert "y = Dataset.read('fred://GDPC1', vintage='latest')" in s.script() and KEY not in s.script()
    b.search("gdp")
    assert codes() == relevance and not header.isSortIndicatorShown()  # a new search: its own order
