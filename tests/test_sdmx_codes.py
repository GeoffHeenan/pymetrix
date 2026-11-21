"""An SDMX dataflow's codes found every way SDMX 2.1 gives them (pymetrix.ext.sdmx_structure), and File → Browse
series…'s pickers, search in a dataflow's codes and hints.

The IMF could not be reached when these were written, so its World Economic Outlook structure is a fixture written by
hand from the SDMX 2.1 standard (tests/fixtures/connectors/search/imf_weo_structure.xml, whose comment lists the cases:
a code list kept by another agency, a concept's representation given by URN, a code list asked for on its own,
content constraints). It is not the IMF's own answer. The expected codes are written out by hand from the fixture.
Nothing here uses the network.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from test_dataflow_groups import _top, browser, imf  # noqa: F401 - the long IMF list, in the browser
from test_series_browser import net  # noqa: F401 - every request answered from a fixture

from pymetrix.commands import LoadFromConnector
from pymetrix.errors import SpecificationError
from pymetrix.ext import connectors as ext
from pymetrix.ext.connector_search import SdmxSearch
from pymetrix.ext.sdmx_structure import Reference, codelists, reference

SEARCH = Path(__file__).parent / "fixtures" / "connectors" / "search"
WEO = (SEARCH / "imf_weo_structure.xml").read_bytes()
FREQ = (SEARCH / "sdmx_cl_freq.xml").read_bytes()
NO_CONSTRAINTS = WEO.split(b"<structure:Constraints>")[0] + WEO.split(b"</structure:Constraints>")[1]

WEO_COUNTRIES = ["USA", "DEU", "JPN", "WSM", "G001"]  # the constraint's, in the code list's order (not FRA, GBR)
WEO_INDICATORS = ["NGDP_RPCH", "NGDPD", "PPPGDP", "PCPIPCH", "LUR", "LP", "GGXCNL_NGDP", "GGXWDG_NGDP", "NGSD_NGDP",
                  "TX_RPCH", "BCA", "BCA_NGDPD"]  # CL_WEO_INDICATOR less NGAP_NPGDP, which the constraint excludes


def _answers(structure: bytes = WEO, codelist: bytes | Exception = FREQ):
    asked: list[str] = []

    def fetch(url: str, kind: str = "metadata") -> bytes:
        asked.append(url)
        if "/dataflow/IMF.RES/WEO/latest?references=all" in url:
            return structure
        if "/codelist/SDMX/CL_FREQ/2.1" in url:
            if isinstance(codelist, Exception):
                raise codelist
            return codelist
        raise AssertionError(f"unexpected URL {url}")

    return fetch, asked


# -- finding the codes ----------------------------------------------------------------------------------------------------


def test_each_dimensions_codes_are_found_however_the_structure_gives_them():
    fetch, asked = _answers()
    s = SdmxSearch().structure("IMF/IMF.RES,WEO", fetch=fetch)
    assert s.title == "World Economic Outlook (WEO)" and [d.id for d in s.dimensions] == ["COUNTRY", "INDICATOR", "FREQUENCY"]
    country, indicator, freq = s.dimensions
    # a code list kept by another agency (IMF.RES, version 2.0, where the dimension names IMF's 1.0), narrowed by the
    # dataflow's constraint; IMF:CL_COUNTRY, a list of the same agency but another id, is not used
    assert [c for c, _ in country.codes] == WEO_COUNTRIES and ("G001", "World") in country.codes
    # the concept's representation (the concept given by URN, in IMF.RES's scheme, not IMF's of the same id)
    assert [c for c, _ in indicator.codes] == WEO_INDICATORS and indicator.name == "Indicator"
    assert dict(indicator.codes)["BCA"] == "Current account balance, U.S. dollars"
    assert dict(indicator.descriptions)["BCA_NGDPD"].startswith("Current account is all transactions")
    assert "PPPGDP" not in dict(indicator.descriptions)  # no description given: none made up
    # a code list missing from the answer, asked for on its own; the constraint allows annual data only (the CPI
    # dataflow's constraint, monthly, is not this dataflow's)
    assert freq.codes == (("A", "Annual"),)
    assert asked == ["https://api.imf.org/external/sdmx/2.1/dataflow/IMF.RES/WEO/latest?references=all",
                     "https://api.imf.org/external/sdmx/2.1/codelist/SDMX/CL_FREQ/2.1"]
    assert all(d.note == "" for d in s.dimensions)
    assert s.address(COUNTRY="USA", INDICATOR=["BCA", "BCA_NGDPD"], FREQUENCY="A") == "sdmx://IMF/IMF.RES,WEO/USA.BCA+BCA_NGDPD.A"
    with pytest.raises(SpecificationError, match="COUNTRY has no code 'FRA'"):  # outside the constraint
        s.key(COUNTRY="FRA")


def test_without_constraints_the_whole_code_lists():
    fetch, _ = _answers(NO_CONSTRAINTS)
    country, indicator, freq = SdmxSearch().structure("IMF/IMF.RES,WEO", fetch=fetch).dimensions
    assert [c for c, _ in country.codes] == ["USA", "DEU", "FRA", "GBR", "JPN", "WSM", "G001"]
    assert "NGAP_NPGDP" in [c for c, _ in indicator.codes]
    assert [c for c, _ in freq.codes] == ["A", "S", "Q", "M", "W", "D"]


def test_codes_that_cannot_be_loaded_are_said_and_can_be_typed():
    # the code list cannot be fetched: the constraint's codes, without names, and a note saying so
    fetch, _ = _answers(codelist=ConnectionError("HTTP Error 403: Forbidden"))
    freq = SdmxSearch().structure("IMF/IMF.RES,WEO", fetch=fetch).dimension("FREQUENCY")
    assert freq.codes == (("A", ""),)
    assert "SDMX:CL_FREQ(2.1) could not be loaded (HTTP Error 403: Forbidden)" in freq.note and "constraint" in freq.note
    # nor any constraint: no codes, and why
    fetch, _ = _answers(NO_CONSTRAINTS, ConnectionError("HTTP Error 403: Forbidden"))
    s = SdmxSearch().structure("IMF/IMF.RES,WEO", fetch=fetch)
    assert s.dimension("FREQUENCY").codes == () and "SDMX:CL_FREQ(2.1) could not be loaded" in s.dimension("FREQUENCY").note
    assert s.key(FREQUENCY="A") == "..A"  # typed codes are taken as given
    # the provider answers the request for the list with an error message, or with no such list
    error = (SEARCH / "sdmx_no_flow.xml").read_bytes()
    fetch, _ = _answers(NO_CONSTRAINTS, error)
    assert "was not in the provider's answers" in SdmxSearch().structure("IMF/IMF.RES,WEO", fetch=fetch).dimension("FREQUENCY").note


def test_references_by_ref_or_urn_and_versions():
    ns = ('xmlns:s="http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure" '
          'xmlns:c="http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common"')
    e = ET.fromstring(f'<s:D {ns}><s:E><Ref id="CL_X" agencyID="IMF" version="1.0" class="Codelist"/></s:E></s:D>')
    assert reference(e, "E") == Reference("IMF", "CL_X", "1.0")
    e = ET.fromstring(f'<s:D {ns}><s:E><URN> urn:sdmx:org.sdmx.infomodel.codelist.Codelist=SDMX:CL_FREQ(2.1) </URN></s:E></s:D>')
    assert reference(e, "E") == Reference("SDMX", "CL_FREQ", "2.1")
    e = ET.fromstring(f'<s:D {ns}><s:E><URN>urn:sdmx:org.sdmx.infomodel.conceptscheme.Concept=IMF.RES:CS_WEO(1.0).INDICATOR</URN></s:E></s:D>')
    assert reference(e, "E") == Reference("IMF.RES", "INDICATOR", "1.0", "CS_WEO")
    e = ET.fromstring(f'<s:D {ns}><s:E><Ref id="FREQ" agencyID="ECB" maintainableParentID="ECB_CONCEPTS" maintainableParentVersion="1.0"/></s:E></s:D>')
    assert reference(e, "E") == Reference("ECB", "FREQ", "1.0", "ECB_CONCEPTS")
    assert reference(e, "Nothing") is None and str(Reference("IMF", "CL_X")) == "IMF:CL_X(latest)"
    lists = codelists(list(ET.fromstring(
        f'<s:L {ns}><s:Codelist id="CL_X" agencyID="IMF" version="1.10"><s:Code id="new"/></s:Codelist>'
        f'<s:Codelist id="CL_X" agencyID="IMF" version="1.9"><s:Code id="old"/></s:Codelist></s:L>').iter()))
    from pymetrix.ext.sdmx_structure import _exact

    assert _exact(lists, Reference("IMF", "CL_X", "1.9"))[0][0] == "old"
    assert _exact(lists, Reference("IMF", "CL_X", "latest"))[0][0] == "new"  # 1.10 after 1.9: versions read as numbers
    assert _exact(lists, Reference("IMF", "CL_X", "2.0")) is None


def test_the_ecbs_structure_is_read_as_before(net):  # noqa: F811
    from pymetrix import connectors as C

    s = C.structure("sdmx", "ECB/EXR")
    assert [d.id for d in s.dimensions] == ["FREQ", "CURRENCY", "CURRENCY_DENOM", "EXR_TYPE", "EXR_SUFFIX"]
    assert all(d.codes and not d.note for d in s.dimensions)
    assert s.dimension("CURRENCY_DENOM").codes == s.dimension("CURRENCY").codes


# -- the browser ---------------------------------------------------------------------------------------------------------


@pytest.fixture()
def weo(browser, monkeypatch):  # noqa: F811
    """The browser on the IMF's list, with WEO's structure answered (and, by default, CL_FREQ)."""
    s, b, asked = browser
    others = ext._get
    answers = {"structure": WEO, "codelist": FREQ}

    def fake(url: str) -> bytes:
        if "/dataflow/IMF.RES/WEO/latest?references=all" in url:
            return answers["structure"]
        if "/codelist/SDMX/CL_FREQ/2.1" in url:
            if isinstance(answers["codelist"], Exception):
                raise answers["codelist"]
            return answers["codelist"]
        return others(url)

    monkeypatch.setattr(ext, "_get", fake)
    return s, b, answers


def _choose_weo(b):
    folder = _top(b, "World Economic Outlook (WEO)")
    row = folder.child(0)
    assert row.text(0) == "World Economic Outlook (WEO)"
    b.tree.itemClicked.emit(row, 0)
    return row


def test_the_series_key_box_says_what_goes_in_it_and_only_the_providers_boxes_show(weo):
    _s, b, _ = weo
    from pymetrix.ui.series_browser_qt import KEY_HINT

    assert b.key_hint is not None and b.key_hint.text() == KEY_HINT == (
        "Choose a dataflow on the left; its dimensions appear here to pick codes from.")
    assert b.key_box.isVisibleTo(b.dialog) and not b.country_box.isVisibleTo(b.dialog)
    assert not b.in_flow.isVisibleTo(b.dialog)  # nothing chosen: the search is of dataflows
    _choose_weo(b)
    assert b.key_hint is None and b.key_box.title() == "Series key: World Economic Outlook (WEO)"
    b.set_provider("worldbank")
    assert b.country_box.isVisibleTo(b.dialog) and not b.key_box.isVisibleTo(b.dialog)
    b.set_provider("fred")
    assert not b.country_box.isVisibleTo(b.dialog) and not b.key_box.isVisibleTo(b.dialog)
    b.set_provider("sdmx")
    assert b.key_hint is not None and b.structure is None


def test_choosing_weo_fills_its_pickers(weo):
    _s, b, _ = weo
    from PySide6 import QtWidgets

    _choose_weo(b)
    assert list(b.pickers) == ["COUNTRY", "INDICATOR", "FREQUENCY"]
    counts = {d: p.count() for d, p in b.pickers.items()}
    assert counts == {"COUNTRY": len(WEO_COUNTRIES), "INDICATOR": len(WEO_INDICATORS), "FREQUENCY": 1}
    assert all(isinstance(p, QtWidgets.QListWidget) for p in b.pickers.values())
    bca = next(b.pickers["INDICATOR"].item(i) for i in range(counts["INDICATOR"]) if b.pickers["INDICATOR"].item(i).data(0x0100) == "BCA")
    assert bca.text() == "BCA  Current account balance, U.S. dollars" and "all transactions other than" in bca.toolTip()
    assert b.key_preview.text() == "sdmx://IMF/IMF.RES,WEO/.." and not b.last_error


def test_searching_inside_the_chosen_dataflow_finds_codes_to_choose(weo):
    _s, b, _ = weo
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from pymetrix.ui.series_browser_qt import CODE_COLUMNS, COLUMNS

    before = b.search("current account")  # nothing chosen: the provider's dataflows, as before
    assert before == [] and b.table.horizontalHeaderItem(0).text() == COLUMNS[0]
    b.query.clear()
    _choose_weo(b)
    assert b.in_flow.isChecked() and b.in_flow.text() == "In WEO's codes" and b.codes_mode()
    from dataclasses import replace

    flow = b.structure
    b.structure = replace(flow, flow="IMF.RES,WEO,1.0")  # AGENCY,ID,VERSION: still named by its id
    assert b._flow_name() == "WEO"
    b.structure = replace(flow, flow="EXR")
    assert b._flow_name() == "EXR"
    b.structure = flow
    assert "WEO's codes" in b.query.placeholderText()
    found = b.search("current account")
    assert [(d, c) for d, c, _n, _t in found] == [("INDICATOR", "BCA"), ("INDICATOR", "BCA_NGDPD"),
                                                  ("INDICATOR", "NGSD_NGDP")]  # names first, then a description's
    assert [b.table.horizontalHeaderItem(i).text() for i in range(b.table.columnCount())] == list(CODE_COLUMNS)
    assert b.table.item(0, 1).text() == "BCA" and b.table.item(0, 2).text() == "INDICATOR"
    assert "double-click" in b.status.text() and "3 of WEO's codes" in b.status.text()
    hidden = [b.tree.topLevelItem(i).isHidden() for i in range(b.tree.topLevelItemCount())]
    assert not any(hidden)  # the words were for the codes: the list of dataflows is not filtered by them
    b.dialog.show()  # mouse and keys reach only what is shown
    QApplication.processEvents()
    view = b.table.viewport()
    at = b.table.visualItemRect(b.table.item(0, 0)).center()
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=at)  # a double-click's first click (Qt 6's QTest sends
    QTest.mouseDClick(view, Qt.MouseButton.LeftButton, pos=at)  # only the second with mouseDClick)
    assert b.choices()["INDICATOR"] == ["BCA"]
    b.table.setFocus()
    b.table.setCurrentCell(1, 0)
    QTest.keyClick(b.table, Qt.Key.Key_Return)  # Enter chooses too
    assert b.choices()["INDICATOR"] == ["BCA", "BCA_NGDPD"]
    assert "Current account is all transactions" in b.details.toPlainText()
    b.choose("COUNTRY", ["USA"])
    b.choose("FREQUENCY", ["A"])
    b.name.setText("ca")
    assert b.command() == LoadFromConnector("sdmx", "IMF/IMF.RES,WEO/USA.BCA+BCA_NGDPD.A", name="ca")
    assert b.search("zzz nothing") == [] and "untick" in b.status.text()
    # untick: the dataflow search again (and the tree filtered by the words, as for dataflows)
    b.in_flow.setChecked(False)
    assert not b.codes_mode() and "unemployment rate" in b.query.placeholderText()
    flows = b.search("labor statistics")
    assert [s.code for s in flows][:1] == ["IMF/IMF.STA,LS_2026_JAN"] and b.table.horizontalHeaderItem(0).text() == "Title"
    assert _top(b, "Consumer Price Index (CPI)").isHidden()
    assert b.structure is not None and b.command().address().startswith("sdmx://IMF/IMF.RES,WEO/")  # the key is kept


def test_codes_that_cannot_be_loaded_are_said_in_the_picker_and_typed(weo):
    _s, b, answers = weo
    from PySide6 import QtWidgets

    answers["structure"] = NO_CONSTRAINTS
    answers["codelist"] = ConnectionError("HTTP Error 403: Forbidden")
    _choose_weo(b)
    typed = b.pickers["FREQUENCY"]
    assert isinstance(typed, QtWidgets.QLineEdit) and typed.placeholderText().startswith("codes, e.g.")
    said = b.unloaded["FREQUENCY"]
    assert list(b.unloaded) == ["FREQUENCY"] and said.isVisibleTo(b.dialog) and said.wordWrap()
    assert said.text().startswith("FREQUENCY: its codes could not be loaded (its code list SDMX:CL_FREQ(2.1) could not be loaded")
    assert "Type codes" in said.text()
    assert "FREQUENCY could not be loaded" in b.last_error
    assert b.pickers["COUNTRY"].count() == 7  # the others are filled
    b.choose("FREQUENCY", ["A", "Q"])
    assert typed.text() == "A+Q" and b.choices()["FREQUENCY"] == ["A", "Q"]
    typed.setText("M")
    b.choose("COUNTRY", ["USA"])
    assert b.key_preview.text() == "sdmx://IMF/IMF.RES,WEO/USA..M"


def test_another_provider_forgets_the_dataflow_chosen(weo):
    _s, b, _ = weo
    _choose_weo(b)
    assert b.in_flow.isChecked()
    b.source.setCurrentIndex(b.source.findText("ECB"))
    assert b.structure is None and not b.pickers and b.key_hint is not None and not b.in_flow.isChecked()
    assert b.tree.topLevelItemCount() == 6  # the ECB's dataflows: no families, a flat list
