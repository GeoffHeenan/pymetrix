"""An SDMX provider's dataflows sorted and grouped by family (pymetrix.ext.dataflow_groups) and File → Browse series…'s
tree of them.

The names are the IMF's, from tests/fixtures/connectors/search/imf_dataflows_long.xml (an IMF dataflow list as long and
unsorted as the IMF gives it, written by hand since the IMF could not be reached), read by the SDMX searcher's own
parser. Nothing here uses the network. The expected orders are written out by hand from the rules; release dates are
checked against the standard library's own reading of them (``datetime.strptime``), not the module's.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_series_browser import net  # noqa: F401 - every request answered from a fixture

from pymetrix import Dataset
from pymetrix import connectors as C
from pymetrix.commands import LoadFromConnector, Session
from pymetrix.ext import connectors as ext
from pymetrix.ext.connector_search import SdmxSearch
from pymetrix.ext.dataflow_groups import group_dataflows, release_date, split_family

FIX = Path(__file__).parent / "fixtures" / "connectors"
LONG = FIX / "search" / "imf_dataflows_long.xml"


def _imf_flows() -> list[tuple[str, str]]:
    flows = SdmxSearch().dataflows("IMF", lambda url, kind: LONG.read_bytes())
    return [(s.code, s.title) for s in flows]


def date_of(text: str) -> datetime:
    return datetime.strptime(text, "%Y %B").replace(tzinfo=UTC)


def _tree(families):
    return [(f.title, [x.label for x in f.flows] if f.folder else None) for f in families]


# -- the grouping ---------------------------------------------------------------------------------------------------------


def test_the_imfs_dataflows_grouped_by_family_and_sorted():
    flows = _imf_flows()
    assert len(flows) == 46 and flows[0][1].startswith("Monetary and Financial Statistics (MFS), Interest Rates 2026")
    families = group_dataflows(flows)
    assert sorted(x.code for f in families for x in f.flows) == sorted(code for code, _ in flows)  # each once
    tops = [f.title for f in families]
    assert tops == sorted(tops, key=str.casefold) and len(tops) == 26
    folders = {title: labels for title, labels in _tree(families) if labels}
    assert folders == {
        "Balance of Payments (BOP)": ["Balance of Payments (BOP)", "World and Regional Aggregates"],
        "Consumer Price Index (CPI)": ["Consumer Price Index (CPI)", "2026 February", "2026 January", "2025 December"],
        "Government Finance Statistics (GFS)": ["Balance Sheet", "Expenditure by Function of Government (COFOG)",
                                                "Statement of Operations"],
        "Labor Statistics (LS)": ["Labor Statistics (LS)", "2026 January", "2025 July"],
        "Monetary and Financial Statistics (MFS)": [
            "Central Bank Survey", "Depository Corporations Survey", "Financial Market Prices", "Interest Rates",
            "Interest Rates 2026 April", "Interest Rates 2026 March", "Monetary Aggregates",
            "Other Depository Corporations Survey", "Other Financial Corporations Survey"],
        "National Economic Accounts (NEA)": ["Annual Data", "Quarterly Data"],
        "World Economic Outlook (WEO)": ["World Economic Outlook (WEO)", "2026 April", "2025 October", "2025 April"],
    }
    # a family of one is a row with its full name, not a folder of one
    rows = {title for title, labels in _tree(families) if labels is None}
    assert {"Fiscal Monitor (FM) 2025 October", "Western Hemisphere Regional Economic Outlook (WHDREO)",
            "Climate Change Indicators", "International Trade in Goods (by partner country) (IMTS)"} <= rows
    fm = next(f for f in families if f.title.startswith("Fiscal Monitor"))
    assert fm.flows[0].label == fm.flows[0].name == "Fiscal Monitor (FM) 2025 October" and fm.flows[0].release == (2025, 10, 0)
    cpi = next(f for f in families if f.title == "Consumer Price Index (CPI)")
    assert [x.code for x in cpi.flows] == ["IMF/IMF.STA,CPI", "IMF/IMF.STA,CPI_2026_FEB", "IMF/IMF.STA,CPI_2026_JAN",
                                           "IMF/IMF.STA,CPI_2025_DEC"]
    assert cpi.flows[1].name == "Consumer Price Index (CPI) 2026 February"  # the full name is kept


def test_release_dates_are_read_and_ordered_as_dates_not_text():
    flows = _imf_flows()
    for family in group_dataflows(flows):
        dated = [x for x in family.flows if x.release]
        for x in dated:
            if x.label != x.name and x.label[:4].isdigit() and len(x.label.split()) == 2:  # a label only a date
                when = date_of(x.label)  # the standard library's reading of it
                assert x.release == (when.year, when.month, 0)
        by_stem: dict[str, list] = {}
        for x in dated:
            by_stem.setdefault(release_date(x.label)[1] if x.label != x.name else "", []).append(x.release)
        for releases in by_stem.values():
            assert releases == sorted(releases, reverse=True)  # newest first
    cpi = ["2026 February", "2026 January", "2025 December"]
    assert sorted(cpi, reverse=True) != cpi  # as text the order would be wrong ("2026 January" > "2026 February")
    assert release_date("Labor Statistics (LS) 2026 January") == ((2026, 1, 0), "Labor Statistics (LS)")
    assert release_date("Survey, January 2026")[0] == (2026, 1, 0)
    assert release_date("Survey 15 Sept 2025")[0] == (2025, 9, 15)
    assert release_date("Survey 2026-03")[0] == (2026, 3, 0)
    assert release_date("Survey 2025Q4")[0] == (2025, 12, 0) and release_date("Survey 2025 Q4")[0] == (2025, 12, 0)
    assert release_date("Survey 2025")[0] == (2025, 0, 0)
    assert release_date("Survey 2025 October")[0] > release_date("Survey 2025")[0]  # a year alone before its months
    for undated in ("Interest Rates", "G20 indicators", "Table 1", "Code 12345", "Index 2010=100", "Survey 2026 Marching"):
        assert release_date(undated) is None, undated


def test_odd_names():
    assert split_family("Labor Statistics (LS)") == ("Labor Statistics (LS)", "")
    assert split_family("Labor Statistics (LS) 2026 January") == ("Labor Statistics (LS)", "2026 January")
    assert split_family("Monetary and Financial Statistics (MFS), Interest Rates") == (
        "Monetary and Financial Statistics (MFS)", "Interest Rates")
    # no parentheses, or none that start like an abbreviation: no family
    assert split_family("Climate Change Indicators") == ("", "Climate Change Indicators")
    assert split_family("HICP - monthly data (annual rate of change)") == ("", "HICP - monthly data (annual rate of change)")
    assert split_family("(MFS) with nothing before it") == ("", "(MFS) with nothing before it")
    assert split_family("Unclosed (MFS, Interest") == ("", "Unclosed (MFS, Interest")
    # nested parentheses and commas inside the abbreviation belong to it
    assert split_family("Direction of Trade Statistics (DOTS (legacy)), Exports") == (
        "Direction of Trade Statistics (DOTS (legacy))", "Exports")
    assert split_family("Government Finance Statistics (GFS, COFOG), Expenditure, by function") == (
        "Government Finance Statistics (GFS, COFOG)", "Expenditure, by function")
    assert split_family("International Trade in Goods (by partner country) (IMTS), Imports") == (
        "International Trade in Goods (by partner country) (IMTS)", "Imports")
    assert split_family("  Labor   Statistics (LS)\t2026 January ") == ("Labor Statistics (LS)", "2026 January")
    odd = [("A/1", "Government Finance Statistics (GFS, COFOG), Expenditure"),
           ("A/2", "Government finance statistics (GFS, COFOG)"),  # the same family whatever the case
           ("A/10", "government finance statistics (gfs, cofog)"),  # no abbreviation: a name of its own, not the family
           ("A/3", "Direction of Trade Statistics (DOTS (legacy)), Exports"),
           ("A/4", "Direction of Trade Statistics (DOTS (legacy)) 2024 May"),
           ("A/5", "Direction of Trade Statistics (DOTS (legacy)), Imports"),
           ("A/6", "No abbreviation here"), ("A/7", "No abbreviation here"),  # the same name twice: two rows
           ("A/8", ""), ("A/9", "zeta (Z)")]
    assert _tree(group_dataflows(odd)) == [
        ("A/8", None),  # no name: its code
        ("Direction of Trade Statistics (DOTS (legacy))", ["2024 May", "Exports", "Imports"]),
        ("Government Finance Statistics (GFS, COFOG)", ["Government finance statistics (GFS, COFOG)", "Expenditure"]),
        ("government finance statistics (gfs, cofog)", None),
        ("No abbreviation here", None), ("No abbreviation here", None),
        ("zeta (Z)", None),
    ]


def test_a_list_with_no_families_is_flat_and_sorted():
    ecb = [(s.code, s.title) for s in SdmxSearch().dataflows("ECB", lambda url, kind: (FIX / "search" / "ecb_dataflows.xml").read_bytes())]
    families = group_dataflows(ecb)
    assert not any(f.folder for f in families) and len(families) == len(ecb)
    assert [f.title for f in families] == sorted((t for _, t in ecb), key=str.casefold)
    assert [f.flows[0].label for f in families] == [f.title for f in families]
    assert group_dataflows([]) == []
    mixed = [("X/b", "beta"), ("X/A", "Alpha"), ("X/c", "Gamma (G)"), ("X/d", "alpha")]
    assert [f.title for f in group_dataflows(mixed)] == ["Alpha", "alpha", "beta", "Gamma (G)"]  # case ignored


# -- the browser ---------------------------------------------------------------------------------------------------------


@pytest.fixture()
def imf(net, monkeypatch):  # noqa: F811
    """The long IMF list, the CPI dataflow's structure and the recorded CPI data; other requests as test_series_browser."""
    others = ext._get

    def fake(url: str) -> bytes:
        if "api.imf.org/external/sdmx/2.1/dataflow/all" in url:
            return LONG.read_bytes()
        if "api.imf.org/external/sdmx/2.1/dataflow/IMF.STA/CPI/latest" in url:
            return (FIX / "search" / "imf_cpi_structure.xml").read_bytes()
        if "api.imf.org/external/sdmx/2.1/data/" in url:
            return (FIX / "imf_cpi.xml").read_bytes()
        return others(url)

    monkeypatch.setattr(ext, "_get", fake)
    return net


@pytest.fixture()
def browser(imf):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.series_browser_qt import series_browser

    s = Session()
    b = series_browser(s)
    b.set_provider("sdmx")
    b.source.setCurrentIndex(b.source.findText("IMF"))
    yield s, b, imf  # imf: the URLs asked for
    b.dialog.close()


def _top(b, title):
    return next(b.tree.topLevelItem(i) for i in range(b.tree.topLevelItemCount()) if b.tree.topLevelItem(i).text(0) == title)


def test_the_browser_shows_the_dataflows_as_a_sorted_tree_with_tooltips(browser):
    _s, b, _asked = browser
    from PySide6.QtCore import Qt

    expected = group_dataflows(_imf_flows())
    tops = [b.tree.topLevelItem(i) for i in range(b.tree.topLevelItemCount())]
    assert [t.text(0) for t in tops] == [f.title for f in expected]
    for item, family in zip(tops, expected):
        assert bool(item.childCount()) == family.folder
        rows = [item.child(j) for j in range(item.childCount())] if family.folder else [item]
        assert [r.text(0) for r in rows] == [x.label for x in family.flows]
        assert [r.data(0, Qt.ItemDataRole.UserRole) for r in rows] == [x.code for x in family.flows]
        for row, flow in zip(rows, family.flows):  # the full name and the dataflow's ID on every row
            assert row.toolTip(0).splitlines()[:2] == [flow.name, f"Dataflow {flow.code.partition('/')[2]}"]
        assert item.toolTip(0)
        if family.folder:
            assert item.toolTip(0) == f"{family.title}\n{len(family.flows)} dataflows"
            assert item.data(0, Qt.ItemDataRole.UserRole) is None  # a folder is not a dataflow
    # every dataflow the provider lists is in the tree once, at the code the flat list had
    listed = {c.path for c in C.browse("sdmx", "IMF").categories}
    codes = [r.data(0, Qt.ItemDataRole.UserRole) for t in tops for r in ([t.child(j) for j in range(t.childCount())] or [t])]
    assert sorted(codes) == sorted(listed) and len(codes) == 46


def test_the_search_words_filter_the_tree_and_open_folders_with_matches(browser):
    _s, b, _asked = browser
    mfs, cpi = _top(b, "Monetary and Financial Statistics (MFS)"), _top(b, "Consumer Price Index (CPI)")
    assert not mfs.isExpanded() and not cpi.isExpanded()
    cpi.setExpanded(True)  # opened by hand: still open after the filter is cleared
    b.query.setText("interest rates")
    shown = [mfs.child(j).text(0) for j in range(mfs.childCount()) if not mfs.child(j).isHidden()]
    assert shown == ["Interest Rates", "Interest Rates 2026 April", "Interest Rates 2026 March"]
    assert mfs.isExpanded() and not mfs.isHidden() and cpi.isHidden()
    visible = [b.tree.topLevelItem(i).text(0) for i in range(b.tree.topLevelItemCount()) if not b.tree.topLevelItem(i).isHidden()]
    assert visible == ["Monetary and Financial Statistics (MFS)"]
    assert b.filter_tree("whdreo") == 1  # by code too; a row of its own
    assert not _top(b, "Western Hemisphere Regional Economic Outlook (WHDREO)").isHidden() and mfs.isHidden()
    assert b.filter_tree("cpi 2026") == 2 and [cpi.child(j).isHidden() for j in range(4)] == [True, False, False, True]
    b.query.setText("")
    assert all(not b.tree.topLevelItem(i).isHidden() for i in range(b.tree.topLevelItemCount()))
    assert all(not mfs.child(j).isHidden() for j in range(mfs.childCount()))
    assert not mfs.isExpanded() and cpi.isExpanded()  # the filter closes what it opened, and only that
    b.query.setText("no such dataflow")
    assert all(b.tree.topLevelItem(i).isHidden() for i in range(b.tree.topLevelItemCount()))
    # the search box still lists what it finds, as before
    found = b.search("labor statistics")
    assert [s.code for s in found] == ["IMF/IMF.STA,LS_2026_JAN", "IMF/IMF.STA,LS", "IMF/IMF.STA,LS_2025_JUL"]  # the provider's order
    assert b.table.rowCount() == 3 and not _top(b, "Labor Statistics (LS)").isHidden()


def test_choosing_a_dataflow_in_a_folder_loads_what_its_address_reads(browser, monkeypatch):
    s, b, asked = browser
    from test_series_browser import _replay_without_the_browser, _same

    cpi = _top(b, "Consumer Price Index (CPI)")
    before = len(asked)
    b._open_item(cpi)  # a folder: nothing is asked for
    assert b.structure is None and len(asked) == before
    b.tree.setCurrentItem(cpi.child(0))
    b.tree.itemClicked.emit(cpi.child(0), 0)  # as a click does
    assert b.structure is not None and b.structure.flow == "IMF.STA,CPI"
    assert list(b.pickers) == ["COUNTRY", "INDEX_TYPE", "COICOP_1999", "TYPE_OF_TRANSFORMATION", "FREQUENCY"]
    for dim, codes in (("COUNTRY", ["USA", "WSM"]), ("INDEX_TYPE", ["CPI"]), ("COICOP_1999", ["CP01"]),
                       ("TYPE_OF_TRANSFORMATION", ["IX"]), ("FREQUENCY", ["M"])):
        b.choose(dim, codes)
    address = "sdmx://IMF/IMF.STA,CPI/USA+WSM.CPI.CP01.IX.M"
    assert b.key_preview.text() == address
    b.name.setText("cpi")
    assert b.command() == LoadFromConnector("sdmx", address.partition("://")[2], name="cpi")
    assert b.load() is not None
    direct = Dataset.read(address)
    _same(s["cpi"], direct)
    assert f"cpi = Dataset.read({address!r}, vintage='latest')" in s.script()
    _same(_replay_without_the_browser(s, "cpi", monkeypatch), direct)


def test_keyboard_moves_through_folders_and_their_dataflows(browser):
    _s, b, _asked = browser
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    cpi = _top(b, "Consumer Price Index (CPI)")
    b.tree.setFocus()
    b.tree.setCurrentItem(cpi)
    QTest.keyClick(b.tree, Qt.Key.Key_Right)  # opens the folder
    assert cpi.isExpanded()
    QTest.keyClick(b.tree, Qt.Key.Key_Down)
    assert b.tree.currentItem() is cpi.child(0)
    QTest.keyClick(b.tree, Qt.Key.Key_Down)
    assert b.tree.currentItem() is cpi.child(1) and b.tree.currentItem().isSelected()
    QTest.keyClick(b.tree, Qt.Key.Key_Left)  # to the folder
    QTest.keyClick(b.tree, Qt.Key.Key_Left)  # and closes it
    assert b.tree.currentItem() is cpi and not cpi.isExpanded()
    assert b.structure is None  # moving does not ask for structures (only a click does, as before)
