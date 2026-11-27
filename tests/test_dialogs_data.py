"""Dialogs for data and nowcasting: Convert frequency, Aggregate over entities (Data), the ragged-edge tools and Combine
nowcasts (Estimate → Mixed frequency), Principal components and Factor analysis (Estimate → Factors), and Load from
connector (File).

Each dialog's run equals the API call it stands for, and its recorded script replays to the same result. The
connector tests answer from the recorded responses in tests/fixtures/connectors: nothing uses the network."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymetrix import PCA, Bridge, Dataset, FactorModel
from pymetrix.commands import (AddToDataset, AggregateEntities, CombineNowcasts, ConvertFrequency, DataAsOf, FillEdge, FitBridge,
                               FitFactorModel, FitPCA, LoadFromConnector, RaggedEdge, RealignEdge, Session)
from pymetrix.nowcast import combine, fill_edge, realign
from pymetrix.testing import check_project

FIXTURES = Path(__file__).parent / "fixtures" / "connectors"


def _replay(s, name, **objects):
    ns = dict(objects)
    exec(s.script(), ns)
    return ns[name]


def _same(a, b):
    pd.testing.assert_frame_equal(a.frame, b.frame)


@pytest.fixture(scope="module")
def data():
    monthly = Dataset.example("sample_monthly")
    quarterly = Dataset.example("sample_macro").with_series(gy="400*dlog(y)")
    growth = monthly.with_series(gr="1200*dlog(retail)", gip="1200*dlog(ip)", gemp="1200*dlog(emp)")
    return {"monthly": monthly, "quarterly": quarterly, "growth": growth, "panel": Dataset.example("grunfeld")}


def _session(data):
    s = Session()
    s.objects.update(data)
    return s


# -- frequency conversion and aggregation --------------------------------------------------------------------------------


@pytest.mark.parametrize("how", ["", "sum", "mean", "first", "last"])
def test_convert_frequency_equals_to_freq(data, how):
    s = _session(data)
    s.run(ConvertFrequency(data="monthly", freq="Q", how=how, name="q"))
    _same(s["q"], data["monthly"].to_freq("Q", how or None))
    assert ("q = monthly.to_freq('Q')" if not how else f"q = monthly.to_freq('Q', how={how!r})") in s.script()
    _same(_replay(s, "q", monthly=data["monthly"]), s["q"])


def test_convert_frequency_rules_years_and_refusals(data, tmp_path):
    s = _session(data)
    s.run(ConvertFrequency(data="monthly", freq="Y-JUN", rules="ip=mean, emp=first", name="fiscal"))
    by_api = data["monthly"].to_freq("Y-JUN", {"ip": "mean", "emp": "first"})
    _same(s["fiscal"], by_api)
    assert s["fiscal"].frame.index.freqstr == "Y-JUN" and by_api["retail"].iloc[1] == pytest.approx(
        data["monthly"]["retail"].loc["2000-07":"2001-06"].sum())  # not in the rules: summed as a flow
    _same(_replay(s, "fiscal", monthly=data["monthly"]), by_api)
    check_project(s, ConvertFrequency(data="monthly", freq="Y", how="last", name="annual"), tmp_path)
    assert ConvertFrequency(data="quarterly", freq="M").validate(s)[0].startswith("quarterly is Q-DEC; M is a higher frequency")
    assert "Temporal disaggregation" in ConvertFrequency(data="quarterly", freq="M").validate(s)[0]
    assert ConvertFrequency(data="quarterly", freq="Q").validate(s) == ["quarterly is already Q-DEC"]
    assert ConvertFrequency(data="monthly", freq="Q", rules="ip=avg").validate(s) == ["ip: 'avg' is not one of sum, mean, first, last, min, max, median"]
    assert ConvertFrequency(data="panel", freq="Q").validate(s)[0].startswith("panel is Y-DEC")
    no_measure = ConvertFrequency(data="growth", freq="Q").validate(s)
    assert len(no_measure) == 1 and "gr, gip, gemp have no measure" in no_measure[0]  # derived series: no measure
    assert ConvertFrequency(data="growth", freq="Q", how="mean").validate(s) == []


def test_aggregate_over_entities(data, tmp_path):
    s = _session(data)
    s.run(AggregateEntities(data="panel", by="year", how="sum", name="total"))
    by_api = data["panel"].aggregate("year", how="sum")
    _same(s["total"], by_api)
    assert by_api["invest"].iloc[0] == pytest.approx(data["panel"].frame.xs(by_api.frame.index[0], level="year")["invest"].sum())
    assert "total = panel.aggregate('year', how='sum')" in s.script()
    _same(_replay(s, "total", panel=data["panel"]), by_api)
    s.run(AggregateEntities(data="panel", by="year", rules="invest=sum, value=mean, capital=max", name="mixed"))
    _same(s["mixed"], data["panel"].aggregate("year", {"invest": "sum", "value": "mean", "capital": "max"}))
    check_project(s, AggregateEntities(data="panel", by="year", how="mean", name="average"), tmp_path)
    assert AggregateEntities(data="panel").validate(s) == ["choose what to keep under By (the time axis, for a total per period)"]
    assert "have no measure" in AggregateEntities(data="panel", by="year").validate(s)[0]
    assert AggregateEntities(data="panel", by="year, region", how="sum").validate(s) == ["no axis or series 'region' in panel"]


# -- the ragged edge and combined nowcasts -------------------------------------------------------------------------------


def test_ragged_edge_and_data_as_of(data, tmp_path):
    s = _session(data)
    s.run(RaggedEdge(data="monthly", when="2025-02-15", release_lags="ip=10"))
    by_api = data["monthly"].ragged_edge("2025-02-15", {"ip": 10})
    pd.testing.assert_frame_equal(s["edge"].to_frame(), by_api.to_frame())
    s.run(DataAsOf("2025-02-15", data="monthly"))
    _same(s["known"], data["monthly"].as_of("2025-02-15"))
    assert "edge = monthly.ragged_edge('2025-02-15', lags={'ip': 10})" in s.script()
    assert "known = monthly.as_of('2025-02-15')" in s.script()
    _same(_replay(s, "known", monthly=data["monthly"]), s["known"])
    check_project(s, DataAsOf("2025-01-31", data="monthly", release_lags="ip=0", name="early"), tmp_path)
    assert DataAsOf("", data="monthly").validate(s) == ["give the day, e.g. 2025-02-15"]
    assert DataAsOf("15 Febtember", data="monthly").validate(s)[0].startswith("the day is a date")
    assert RaggedEdge(data="monthly", release_lags="ip=-3, gdp=1").validate(s) == [
        "ip: a publication lag is days after the period ends, 0 or more", "no series 'gdp' in the data"]
    assert RaggedEdge(data="panel").validate(s)[0].startswith("panel is not a time series")


def test_fill_and_realign_the_edge(data, tmp_path):
    s = _session(data)
    s.objects["known"] = data["monthly"].as_of("2025-02-15")
    s.run(FillEdge(data="known", lags=3, through="2025-03", series="ip, pmi"))
    by_api = fill_edge(s["known"], lags=3, through="2025-03", names=["ip", "pmi"])
    _same(s["filled"], by_api)
    assert s["filled"]["retail"].loc["2025-03"] != s["filled"]["retail"].loc["2025-03"]  # not chosen: left missing (NaN)
    s.run(RealignEdge(data="known"))
    _same(s["realigned"], realign(s["known"]))
    script = s.script()
    assert "filled = known.fill_edge(lags=3, through='2025-03', names=['ip', 'pmi'])" in script and "realigned = known.realign()" in script
    _same(_replay(s, "filled", known=s["known"]), by_api)
    check_project(s, FillEdge(data="known", name="filled2"), tmp_path)
    assert FillEdge(data="known", series="gdp").validate(s) == ["no series 'gdp' in known"]


def test_combine_nowcasts(data, tmp_path):
    s = _session(data)
    s.objects["known"] = data["growth"].as_of("2025-02-15")
    s.run(FitBridge("gy ~ gr + pmi", low="quarterly", high="growth", name="b1"))
    s.run(FitBridge("gy ~ gip + gemp", low="quarterly", high="growth", name="b2"))
    s.run(CombineNowcasts("b1, b2", low="quarterly", high="known", weights="b1=3, b2=1"))
    one = Bridge("gy ~ gr + pmi").fit(data["quarterly"], data["growth"]).nowcast(data["quarterly"], s["known"])
    two = Bridge("gy ~ gip + gemp").fit(data["quarterly"], data["growth"]).nowcast(data["quarterly"], s["known"])
    by_api = combine({"b1": one, "b2": two}, {"b1": 3, "b2": 1})
    pd.testing.assert_series_equal(s["combined"], by_api)
    assert s["combined"].loc["2025Q1"] == pytest.approx(0.75 * one.loc["2025Q1"] + 0.25 * two.loc["2025Q1"])
    assert ("combined = nowcast.combine({'b1': b1.nowcast(quarterly, known), 'b2': b2.nowcast(quarterly, known)}, "
            "weights={'b1': 3.0, 'b2': 1.0})") in s.script()
    pd.testing.assert_series_equal(_replay(s, "combined", quarterly=data["quarterly"], growth=data["growth"], known=s["known"]), by_api)
    check_project(s, CombineNowcasts("b1, b2", low="quarterly", high="known", name="equal"), tmp_path)
    assert CombineNowcasts("").validate(s) == ["choose the models whose nowcasts to combine"]
    assert CombineNowcasts("b1, quarterly", low="quarterly", high="known").validate(s) == [
        "quarterly is not a bridge equation or a dynamic factor model"]
    assert CombineNowcasts("b1", high="known").validate(s) == ["choose the quarterly data for the bridge equations"]
    assert CombineNowcasts("b1, b2", low="quarterly", high="known", weights="b3=1, b1=-1").validate(s) == [
        "b3 is not one of the models chosen", "weights must be 0 or more, and not all 0"]


def test_combine_a_bridge_with_a_dynamic_factor_model(data):
    pytest.importorskip("statsmodels")
    from pymetrix import DynamicFactor
    from pymetrix.commands import FitDynamicFactor

    s = _session(data)
    s.objects["known"] = data["growth"].as_of("2025-02-15")
    s.run(FitBridge("gy ~ gr + pmi", low="quarterly", high="growth"))
    s.run(FitDynamicFactor("gr, gip, gemp, pmi", high="known", quarterly="gy", low="quarterly"))
    s.run(CombineNowcasts("bridge, dfm", variable="gy", low="quarterly", high="known"))
    dfm = DynamicFactor(monthly=["gr", "gip", "gemp", "pmi"], quarterly=["gy"]).fit(data["quarterly"], s["known"])
    bridge = Bridge("gy ~ gr + pmi").fit(data["quarterly"], data["growth"])
    by_api = combine({"bridge": bridge.nowcast(data["quarterly"], s["known"]), "dfm": dfm.nowcast("gy")})
    pd.testing.assert_series_equal(s["combined"], by_api, rtol=1e-8)
    assert "'dfm': dfm.nowcast('gy')" in s.script()
    assert CombineNowcasts("dfm").validate(s) == ["name the series dfm nowcasts (Variable)"]
    assert CombineNowcasts("dfm", variable="gdp").validate(s)[0].startswith("dfm has no series 'gdp'")


# -- principal components and factor analysis ----------------------------------------------------------------------------


def test_principal_components(data, tmp_path):
    s = _session(data)
    names = ["gr", "gip", "gemp", "pmi"]
    s.run(FitPCA("gr, gip, gemp, pmi", data="growth", n=2, start="2005-01", end="2024-12"))
    by_api = PCA(names, n=2).fit(data["growth"], span=("2005-01", "2024-12"))
    pd.testing.assert_frame_equal(s["pca"].loadings, by_api.loadings)
    pd.testing.assert_frame_equal(s["pca"].scores, by_api.scores)
    assert "pca = PCA(variables=['gr', 'gip', 'gemp', 'pmi'], n=2).fit(growth, span=('2005-01', '2024-12'))" in s.script()
    pd.testing.assert_frame_equal(_replay(s, "pca", growth=data["growth"]).loadings, by_api.loadings)
    s.run(FitPCA("gr, gip, gemp, pmi", data="growth", standardize=False, name="cov"))
    pd.testing.assert_series_equal(s["cov"].eigenvalues, PCA(names, standardize=False).fit(data["growth"]).eigenvalues)
    s.run(AddToDataset("pca", "scores:pc1", data="growth", series_name="pc1"))  # Next ▸ Add components to a dataset
    np.testing.assert_allclose(s["growth"]["pc1"].dropna().to_numpy(), by_api.scores["pc1"].to_numpy())
    assert "growth = growth.with_series(pc1=pca.scores['pc1'])" in s.script()
    check_project(s, FitPCA("gr, gip, pmi", data="growth", name="three"), tmp_path)
    assert FitPCA("gr", data="growth").validate(s) == ["choose at least 2 series"]
    assert FitPCA("gr, gip", data="growth", n=3).validate(s) == ["at most 2 components: one per series"]


@pytest.mark.parametrize("rotation", ["none", "varimax", "promax"])
def test_factor_analysis(rotation, tmp_path):
    mroz = Dataset.example("mroz")
    names = ["educ", "motheduc", "fatheduc", "age", "exper"]
    s = Session()
    s.objects["mroz"] = mroz
    s.run(FitFactorModel(", ".join(names), data="mroz", factors=2, rotation=rotation))
    by_api = FactorModel(names, factors=2, rotation=rotation).fit(mroz)
    pd.testing.assert_frame_equal(s["fa"].loadings, by_api.loadings)
    assert s["fa"].test.p_value == pytest.approx(by_api.test.p_value)  # the number-of-factors test
    pd.testing.assert_frame_equal(s["fa"].scores("bartlett"), by_api.scores("bartlett"))
    rotated = "" if rotation == "none" else f", rotation={rotation!r}"
    assert f"fa = FactorModel(variables=['educ', 'motheduc', 'fatheduc', 'age', 'exper'], factors=2{rotated}).fit(mroz)" in s.script()
    assert "from pymetrix_finance" not in s.script()  # the finance plugin's FactorModel result does not displace the core's
    s.run(AddToDataset("fa", "bartlett:factor1", data="mroz"))  # Next ▸ Add factor scores to a dataset
    np.testing.assert_allclose(s["mroz"]["fa_bartlett_factor1"].dropna().to_numpy(), by_api.scores("bartlett")["factor1"].to_numpy())
    replayed = _replay(s, "fa", mroz=mroz)
    pd.testing.assert_frame_equal(replayed.loadings, by_api.loadings)
    if rotation == "promax":
        check_project(s, FitFactorModel(", ".join(names), data="mroz", factors=1, name="one"), tmp_path)
    assert FitFactorModel("educ, age", data="mroz").validate(s) == ["choose at least 3 series"]
    assert FitFactorModel("educ, age, exper", data="mroz", factors=2).validate(s)[0].startswith("2 factors are too many for 3 series")


# -- data connectors -----------------------------------------------------------------------------------------------------


@pytest.fixture()
def responses(monkeypatch):
    """Recorded responses, as tests/test_connectors.py uses them; the URLs asked for are kept."""
    from pymetrix.ext import connectors

    asked = []
    routes = {"fredgraph.csv?id=GDPC1&vintage_date": "fred_gdpc1_2020.csv", "fredgraph.csv?id=GDPC1": "fred_gdpc1.csv",
              "worldbank.org/v2/country/WSM;TON/indicator/NY.GDP.MKTP.CD": None, "data-api.ecb.europa.eu": "ecb_exr.csv"}

    def fake(url):
        asked.append(url)
        for pattern, name in routes.items():
            if pattern in url:
                name = name or ("wb_page2.json" if "page=2" in url else "wb_page1.json")
                return (FIXTURES / name).read_bytes()
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(connectors, "_get", fake)
    # A recorded response was retrieved once, at one time. Without this, reopening a project re-fetches its "latest"
    # loads and stamps them with the clock's second then: the data dictionary's "retrieved" column differs from the
    # first run whenever a second ticks in between (on a slow CI machine, often), and check_project sees a change.
    monkeypatch.setattr(connectors, "_now", lambda: "2026-01-15T12:00:00")
    return asked


def test_load_from_fred_with_periods_and_a_vintage(responses, tmp_path):
    s = Session()
    assert {"fred", "worldbank", "sdmx"} <= set(s.choices(next(f for f in __import__("pymetrix").commands.form_spec(LoadFromConnector)
                                                              if f.name == "provider"), {}))
    cmd = LoadFromConnector("fred", "GDPC1", start="1947Q3", end="1948Q2", vintage="2020-01-30", name="gdp")
    assert cmd.validate(s) == [] and cmd.address() == "fred://GDPC1?start=1947Q3&end=1948Q2"
    s.run(cmd)
    by_api = Dataset.read("fred://GDPC1?start=1947Q3&end=1948Q2", vintage="2020-01-30")
    _same(s["gdp"], by_api)
    assert str(by_api.frame.index[0]) == "1947Q3" and str(by_api.frame.index[-1]) == "1948Q2"
    assert "vintage_date=2020-01-30" in responses[-1] and s["gdp"].meta.vintage == {"policy": "as_of", "date": "2020-01-30"}
    assert "gdp = Dataset.read('fred://GDPC1?start=1947Q3&end=1948Q2', vintage='2020-01-30')" in s.script()
    _same(_replay(s, "gdp"), by_api)
    s.run(LoadFromConnector("fred", "GDPC1", name="latest"))
    assert "latest = Dataset.read('fred://GDPC1', vintage='latest')" in s.script()  # the request is written out
    _same(s["latest"], Dataset.read("fred://GDPC1"))
    check_project(s, LoadFromConnector("fred", "GDPC1", start="1948Q1", name="recent"), tmp_path)


def test_projects_keep_connector_data_as_read_and_refresh_reads_it_again(responses, monkeypatch, tmp_path):
    """A dataset from Load from connector is saved in the project: opening it does not download again, so a revision
    at the source does not change the project's numbers. Refreshing reads the source again, re-runs the steps that
    changed the dataset in place, and recomputes what was computed from it; the project then keeps the new data."""
    from pymetrix.commands import GenerateSeries, load_project, save_project
    from pymetrix.ext import connectors

    s = Session()
    s.run(LoadFromConnector("fred", "GDPC1", name="gdp"))
    s.run(GenerateSeries("g", "400*dlog(gdpc1)", data="gdp"))
    assert s.refreshable() == ["gdp"]
    first = s["gdp"].frame.copy()
    path = save_project(s, tmp_path / "gdp.pymx")

    original = (FIXTURES / "fred_gdpc1.csv").read_text()
    lines = original.splitlines()
    revised = "\n".join(lines[:-1] + [lines[-1].split(",")[0] + ",9999.0"]) + "\n"  # the last quarter revised
    asked = len(responses)
    monkeypatch.setattr(connectors, "_get", lambda url: (responses.append(url), revised.encode())[1])
    reopened = load_project(path)
    assert len(responses) == asked  # nothing downloaded on opening
    pd.testing.assert_frame_equal(reopened["gdp"].frame, first)

    done = reopened.refresh()
    assert len(responses) == asked + 1 and [type(r.command).__name__ for r in done] == ["LoadFromConnector", "GenerateSeries"]
    frame = reopened["gdp"].frame
    assert frame["gdpc1"].iloc[-1] == 9999.0 and frame["gdpc1"].iloc[:-1].equals(first["gdpc1"].iloc[:-1])
    np.testing.assert_allclose(frame["g"].iloc[-1], 400 * np.log(9999.0 / first["gdpc1"].iloc[-2]))  # generated again
    again = load_project(save_project(reopened, tmp_path / "refreshed.pymx"))
    assert len(responses) == asked + 1  # the refreshed data is what the project keeps
    pd.testing.assert_frame_equal(again["gdp"].frame, frame)
    with pytest.raises(Exception, match="cannot be refreshed"):
        again.refresh("nosuch")


def test_refresh_refuses_data_changed_by_typed_code(responses):
    s = Session()
    s.run(LoadFromConnector("fred", "GDPC1", name="gdp"))
    s.run_code("gdp = gdp.with_series(two=2 * gdp['gdpc1'])")
    with pytest.raises(Exception, match="changed by typed code"):
        s.refresh("gdp")


def test_load_from_the_world_bank_and_sdmx(responses):
    s = Session()
    s.run(LoadFromConnector("worldbank", "NY.GDP.MKTP.CD", start="2021", end="2023", options="country=WSM;TON&per_page=3", name="wb"))
    assert "date=2021%3A2023" in responses[-1] or "date=2021:2023" in responses[-1]
    _same(s["wb"], Dataset.read("worldbank://NY.GDP.MKTP.CD?country=WSM;TON&date=2021:2023&per_page=3"))
    assert s["wb"].kind == "panel"
    s.run(LoadFromConnector("sdmx", "ECB/EXR/M.USD+GBP.EUR.SP00.A", start="2024-01", end="2024-03", name="fx"))
    assert "startPeriod=2024-01" in responses[-1] and "endPeriod=2024-03" in responses[-1]
    _same(s["fx"], Dataset.read("sdmx://ECB/EXR/M.USD+GBP.EUR.SP00.A?startPeriod=2024-01&endPeriod=2024-03"))
    assert "fx = Dataset.read('sdmx://ECB/EXR/M.USD+GBP.EUR.SP00.A?startPeriod=2024-01&endPeriod=2024-03', vintage='latest')" in s.script()
    _same(_replay(s, "fx"), s["fx"])
    assert LoadFromConnector("sdmx", "ECB/EXR/M.USD.EUR.SP00.A", vintage="2020-01-30").validate(s) == [
        "the sdmx connector keeps no vintages: leave Vintage empty for the latest data"]
    assert LoadFromConnector("fred", "").validate(s) == ["give the series codes (series IDs, e.g. GDPC1,UNRATE)"]
    assert LoadFromConnector("fred", "GDPC1", vintage="last week").validate(s) == ["the vintage is a date such as 2020-01-30, not 'last week'"]
    assert LoadFromConnector("nosuch", "X").validate(s)[0].startswith("no connector 'nosuch'; installed: ")
    assert LoadFromConnector("fred", "GDPC1?vintage=2020-01-30").validate(s) == ["give other parameters under Options, not after ? in the codes"]


# -- the forms, menus and next steps -------------------------------------------------------------------------------------


def test_menus_next_steps_and_help():
    from pymetrix.commands import COMMANDS
    from pymetrix.commands.menus import menu_order
    from pymetrix.help import topic
    from pymetrix.ui.next_steps import steps_for

    paths = {info.cls.__name__: path for info, path in menu_order(list(COMMANDS.values()))}
    assert paths["ConvertFrequency"] == "Data → Transform → Convert frequency…"
    assert paths["AggregateEntities"] == "Data → Panel → Aggregate over entities…"
    assert paths["LoadFromConnector"] == "File → Load from connector…"
    for name, label in (("RaggedEdge", "Ragged edge…"), ("DataAsOf", "Data as known on a day…"), ("FillEdge", "Fill the ragged edge…"),
                        ("RealignEdge", "Realign the ragged edge…"), ("CombineNowcasts", "Combine nowcasts…")):
        assert paths[name] == f"Estimate → Mixed frequency → {label}"
    assert paths["FitPCA"] == "Estimate → Factors → Principal components…"
    assert paths["FitFactorModel"] == "Estimate → Factors → Factor analysis…"
    order = list(paths)
    assert order.index("LoadFromConnector") == order.index("LoadData") + 1
    for t in ("frequency-conversion", "nowcasting", "factor-analysis", "data-connectors"):
        assert topic(t).text
    fa = FactorModel(["educ", "motheduc", "fatheduc"], factors=1).fit(Dataset.example("mroz"))
    assert [(label, cls.__name__, field) for label, cls, field, _ in steps_for(fa)] == [
        ("Add factor scores to a dataset…", "AddToDataset", "source")]


def test_the_qt_forms(data, responses):
    pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import command_dialog

    s = _session(data)
    convert = command_dialog(ConvertFrequency, s)
    assert convert.on_advanced_tab("rules") and convert.row_visible("rules")
    convert.editors["how"].set("sum")
    assert not convert.row_visible("rules")  # rules only when converting by measure
    connector = command_dialog(LoadFromConnector, s)
    assert connector.row_visible("vintage") and connector.row_visible("start") and connector.on_advanced_tab("options")
    connector.editors["provider"].set("sdmx")
    assert not connector.row_visible("vintage") and connector.row_visible("start")
    assert command_dialog(DataAsOf, s).on_advanced_tab("release_lags")
    assert command_dialog(CombineNowcasts, s).on_advanced_tab("through")
    for cls in (AggregateEntities, RaggedEdge, FillEdge, RealignEdge, FitPCA, FitFactorModel):
        assert command_dialog(cls, s) is not None


def test_file_refresh_data_reads_the_checked_datasets_again(responses, monkeypatch):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui import qt as ui

    def no_box(*args, **kwargs):  # a message box would wait for a click
        raise AssertionError(f"a message box: {args[1:]}")

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", no_box)
    s = Session()
    s.run(LoadFromConnector("fred", "GDPC1", name="gdp"))
    window = ui.main_window(s)
    try:
        file_menu = window.command_menus["File"]
        labels = [a.text().replace("&", "") for a in file_menu.actions()]
        assert labels.index("Refresh data…") == labels.index("Browse series…") + 1
        asked = len(responses)
        next(a for a in file_menu.actions() if a.text().replace("&", "") == "Refresh data…").trigger()
        dialog = window.refresh_data_dialog
        assert dialog.listing.count() == 1 and "fred://GDPC1" in dialog.listing.item(0).text()
        assert dialog.chosen() == ["gdp"]
        dialog.accept()
        session = window.state["session"]
        assert len(responses) == asked + 1 and [type(r.command).__name__ for r in session.history] == ["LoadFromConnector"] * 2
    finally:
        window.state["session"].dirty = False  # closing would ask to save
        window.close()
