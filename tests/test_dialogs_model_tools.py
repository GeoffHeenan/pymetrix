"""The model dialogs: Model > Shock decomposition, Model > Export model, Model > Import > FRB/US XML file and EViews
state-space object, Model > Add factors, Model > Policy > Optimal control, and Model > Solve model's stochastic simulation
from a sample of residuals (errors: sample, error scale, workers) with add factors in its scenario.

Each dialog's run equals the API call it stands for, and the script the session records runs to the same result. The
FRB/US checks use the small excerpt in tests/fixtures/eviews, not the Fed's packages."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Model, Scenario, export_model, optimal_control, templates
from pymetrix.commands import (AddFactors, ExportModel, ImportEViewsStateSpace, ImportFRBUS, LoadTemplate, OptimalControlCommand,
                               Session, ShockDecomposition, SolveModel)
from pymetrix.commands.builtin import EstimateModel
from pymetrix.models.eviews import read_eviews_sspace, read_eviews_xml
from pymetrix.views import view_sections

FIXTURES = Path(__file__).parent / "fixtures" / "eviews"
XML = FIXTURES / "frbus_excerpt.xml"

SSPACE = """' a trend and a cycle, as an EViews state-space object
sspace ss
ss.append @signal y = 0.5*x + trend + cycle + [var = c(3)^2]
ss.append @state trend = trend(-1) + [var = c(1)^2]
ss.append @state cycle = c(4)*cycle(-1) + c(5)*cycle1(-1) + ecyc
ss.append @state cycle1 = cycle(-1)
ss.append @ename ecyc
ss.append @evar var(ecyc) = c(2)^2
c(1) = 0.3
c(2) = 0.8
c(3) = 0.5
c(4) = 1.2
c(5) = -0.4
"""


def _replay(s, name, extra=None):
    ns = {**dict(s.objects), **(extra or {})}
    exec(s.script(), ns)
    return ns[name]


def _frbus_data(model):
    frame = pd.read_csv(FIXTURES / "frbus_longbase_excerpt.csv", comment="#")
    frame.index = pd.PeriodIndex(frame.pop("obs"), freq="Q", name="time")
    missing = [v for v in model.exogenous if v.endswith("_aerr") and v not in frame.columns]  # the add factor series, at zero
    return Dataset.from_frame(pd.concat([frame, pd.DataFrame(0.0, index=frame.index, columns=missing)], axis=1))


@pytest.fixture(scope="module")
def estimated():
    nk = templates.new_keynesian()
    truth = nk.solve().simulate(80, nk.shock_sd, seed=4)
    truth.index = pd.period_range("2000Q1", periods=80, freq="Q", name="time")
    s = Session()
    s.objects["nk"] = nk
    s.objects["data"] = nk.measurement.apply(Dataset.from_frame(truth))
    s.run(EstimateModel("nk", estimate="rho"))
    return s


def test_shock_decomposition_of_an_estimated_model(estimated):
    s = estimated
    s.run(ShockDecomposition("estimated"))
    s.run(ShockDecomposition("estimated", start="presample", name="from_presample"))
    fit = s["estimated"]
    for name, start in (("shock_decomposition", "first"), ("from_presample", "presample")):
        by_api = fit.decompose(start=start)
        for v in by_api.contributions:
            pd.testing.assert_frame_equal(s[name].contributions[v], by_api.contributions[v])
            pd.testing.assert_frame_equal(_replay(s, name).contributions[v], by_api.contributions[v])
    script = s.script()
    assert "shock_decomposition = estimated.decompose()" in script
    assert "from_presample = estimated.decompose('presample')" in script
    assert ShockDecomposition("estimated", start="presample", name="from_presample").to_code() in script
    assert ShockDecomposition("estimated", start="presample", occbin=True).validate(s) == [
        "OccBin decompositions start from the first period's state: choose start first"]
    assert ShockDecomposition("nope").validate(s) == ["no estimated model named 'nope'"]
    assert any(t.startswith("Table") for t, _ in view_sections(s["shock_decomposition"]))


def test_export_model_with_its_notes():
    s = Session()
    s.run(LoadTemplate("new_keynesian", name="nk"))
    s.run(ExportModel("nk", "dynare", name="mod"))
    s.run(ExportModel("nk", "eviews", form="text", origin="the template", name="prg"))
    nk = s["nk"]
    for name, by_api in (("mod", export_model(nk, "dynare")), ("prg", export_model(nk, "eviews", origin="the template", form="text"))):
        assert s[name].text == by_api.text and s[name].notes == by_api.notes and s[name].checked == by_api.checked
        assert _replay(s, name).text == by_api.text
    assert "mod = export_model(nk, 'dynare')" in s.script()
    assert "form='text'" in s.script()
    assert ExportModel("nk", "iris", form="text").to_call().arguments["form"] == "program"  # form is EViews's alone
    tabs = dict(view_sections(s["mod"]))
    assert list(tabs) == ["Notes", "Text"] and tabs["Text"].rows[0][1] == s["mod"].text.splitlines()[0]


def test_import_frbus_xml_and_an_eviews_state_space(tmp_path):
    s = Session()
    s.run(ImportFRBUS(str(XML)))
    s.run(ImportFRBUS(str(XML), mce="mcap", name="frbus_mce"))
    for name, mce in (("frbus", False), ("frbus_mce", ["mcap"])):
        by_api = read_eviews_xml(XML, mce=mce)
        assert s[name].equations == by_api.equations and s[name].model.endogenous == by_api.model.endogenous
        assert _replay(s, name).equations == by_api.equations
    assert "frbus_mce = read_eviews_xml(" in s.script() and "mce=['mcap']" in s.script()
    assert [t for t, _ in view_sections(s["frbus"])] == ["Variables", "Equations", "Parameters"]

    path = tmp_path / "trend_cycle.prg"
    path.write_text(SSPACE, encoding="utf-8")
    s.run(ImportEViewsStateSpace(str(path), params="c(4)=1.1, c_5=-0.3"))
    by_api = read_eviews_sspace(str(path), params={"c_4": 1.1, "c_5": -0.3})
    assert repr(s["sspace"]) == repr(by_api) and repr(_replay(s, "sspace")) == repr(by_api)
    assert "read_eviews_sspace(" in s.script() and "'c_4': 1.1" in s.script()
    assert {p.name: p.start for p in s["sspace"].params}["c_4"] == 1.1
    assert [t for t, _ in view_sections(s["sspace"])] == ["Equations", "Parameters"]
    assert ImportEViewsStateSpace(str(path), params="c(4)").validate(s)


@pytest.fixture(scope="module")
def frbus():
    read = read_eviews_xml(XML)
    s = Session()
    s.objects["frbus"] = read
    s.objects["data"] = _frbus_data(read.model)
    return s


def test_add_factors_make_the_model_track_the_data(frbus):
    s = frbus
    s.run(AddFactors("frbus", start="2026Q2", end="2027Q2", name="track"))
    model, data = s["frbus"].model, s["data"]
    by_api = model.add_factors(data, ("2026Q2", "2027Q2"))
    assert set(s["track"]) == set(by_api)
    for k in by_api:
        pd.testing.assert_series_equal(s["track"][k], by_api[k])
        pd.testing.assert_series_equal(_replay(s, "track")[k], by_api[k])
    assert "track = Model([frbus]).add_factors(data, span=('2026Q2', '2027Q2'))" in s.script()
    assert [t for t, _ in view_sections(s["track"])] == ["Values"]
    # Solve with them (Scenario, Add factors) reproduces the data
    blank = data.frame.copy()
    blank.loc["2026Q2":"2027Q2", list(model.endogenous)] = np.nan
    s.objects["blank"] = Dataset.from_frame(blank)
    s.run(SolveModel("frbus", "2026Q2", "2027Q2", data="blank", adjust="track", name="tracked"))
    direct = model.solve(s["blank"], ("2026Q2", "2027Q2"), Scenario(adjust=by_api))
    for v in model.endogenous:
        np.testing.assert_allclose(s["tracked"][v].loc["2026Q2":"2027Q2"], data[v].loc["2026Q2":"2027Q2"], atol=1e-8)
        np.testing.assert_allclose(_replay(s, "tracked")[v].loc["2026Q2":"2027Q2"], direct[v].loc["2026Q2":"2027Q2"], atol=1e-12)
    assert AddFactors("").validate(s) == ["choose the model's equations"]
    assert SolveModel("frbus", "2026Q2", "2027Q2", adjust="nothing").validate(s) == ["no add factors named 'nothing'"]


def test_solve_resamples_a_residual_sample(frbus):
    s = frbus
    model, data = s["frbus"].model, s["data"]
    history = model.add_factors(data, ("2025Q2", "2026Q4"))
    residuals = pd.DataFrame(history).dropna()
    s.objects["resid"] = Dataset.from_frame(pd.concat([residuals, pd.DataFrame({"other": 1.0}, index=residuals.index)], axis=1)
                                            .reindex(data.frame.index))
    s.objects["history"] = history
    span = ("2027Q1", "2027Q4")
    first, last = str(residuals.index[0]), str(residuals.index[-1])
    s.run(SolveModel("frbus", *span, draws=30, seed=3, errors="sample", residuals="resid", residual_series=", ".join(residuals.columns),
                     residual_start=first, residual_end=last, error_scale="2027Q1=0.5", workers=1, name="sampled"))
    s.run(SolveModel("frbus", *span, draws=30, seed=3, errors="sample", residuals="history", name="from_factors"))
    direct = model.solve(data, span, draws=30, seed=3, errors=residuals, error_scale={"2027Q1": 0.5}, workers=1)
    np.testing.assert_allclose(s["sampled"].simulation.values, direct.simulation.values, equal_nan=True)
    np.testing.assert_allclose(_replay(s, "sampled").simulation.values, direct.simulation.values, equal_nan=True)
    plain = model.solve(data, span, draws=30, seed=3, errors=residuals)
    np.testing.assert_allclose(s["from_factors"].simulation.values, plain.simulation.values, equal_nan=True)
    np.testing.assert_allclose(_replay(s, "from_factors").simulation.values, plain.simulation.values, equal_nan=True)
    script = s.script()
    assert "errors=resid.keep(" in script and ".over(" in script and "error_scale={'2027Q1': 0.5}" in script
    assert "errors=history" in script
    bad = SolveModel("frbus", *span, draws=30, errors="sample")
    assert bad.validate(s) == ["errors: sample: choose the residuals to resample"]
    assert SolveModel("frbus", *span, draws=30, errors="sample", residuals="history", residual_start=first).validate(s)
    assert SolveModel("frbus", *span, draws=30, error_scale="2027Q1").validate(s)[0].startswith("stochastic simulation:")


def test_optimal_control_of_a_backward_looking_model(frbus):
    s = frbus
    model, data = s["frbus"].model, s["data"]
    track = model.add_factors(data, ("2026Q2", "2027Q4"))
    s.objects["track_oc"] = track
    s.run(OptimalControlCommand("frbus", "2026Q2", "2027Q4", "rff", "zdivgr=1, delrff=1", desired="zdivgr=0.05",
                                control_end="2027Q2", discount=0.99, adjust="track_oc", name="oc"))
    by_api = optimal_control(model, data, ("2026Q2", "2027Q4"), ["rff"], {"zdivgr": 1.0, "delrff": 1.0}, desired={"zdivgr": 0.05},
                             control=("2026Q2", "2027Q2"), discount=0.99, scenario=Scenario(adjust=track))
    pd.testing.assert_frame_equal(s["oc"].moves, by_api.moves)
    assert s["oc"].loss == pytest.approx(by_api.loss, rel=1e-12)
    replayed = _replay(s, "oc")
    pd.testing.assert_frame_equal(replayed.moves, by_api.moves)
    script = s.script()
    assert "oc = optimal_control(Model([frbus]), data, ('2026Q2', '2027Q4'), ['rff'], {'zdivgr': 1.0, 'delrff': 1.0}" in script
    assert "Scenario(adjust=track_oc)" in script
    assert [t for t, _ in view_sections(s["oc"])] == ["Summary", "Moves", "Paths", "Gradient"]
    assert OptimalControlCommand("frbus", "2026Q2", "2027Q4", "rff", "").validate(s) == ["give the loss weights, e.g. ugap=1, pic4=1"]
    assert OptimalControlCommand("frbus", "2026Q2", "2027Q4", "rff", "zdivgr=1", desired="rff=1").validate(s) == [
        "desired values for rff, which are not in the loss"]


def test_a_model_built_from_a_model_read():
    read = read_eviews_xml(XML)
    combined = Model([read], params={"y_zyh_1": 0.5})
    assert combined.endogenous == read.model.endogenous
    assert combined.parameters["y_zyh_1"].value == 0.5
    assert combined.parameters["y_egfe_1"].value == read.model.parameters["y_egfe_1"].value


def test_the_dialogs_in_the_menus_and_next_steps(estimated):
    from pymetrix.commands import COMMANDS
    from pymetrix.commands.menus import menu_order
    from pymetrix.ui.next_steps import steps_for

    paths = {i.cls.__name__: p for i, p in menu_order([i for i in COMMANDS.values() if i.menu])}
    assert paths["ShockDecomposition"] == "Model → Analysis → Shock decomposition…"
    assert paths["ImportFRBUS"] == "Model → Import → FRB/US XML file…"
    assert paths["ImportEViewsStateSpace"] == "Model → Import → EViews state-space object…"
    assert paths["ExportModel"] == "Model → Export model…" and paths["AddFactors"] == "Model → Solution → Add factors…"
    assert paths["OptimalControlCommand"] == "Model → Policy → Optimal control…"
    assert "ShockDecomposition" in [c.__name__ for _, c, _, _ in steps_for(estimated["estimated"])]
    assert [c.__name__ for _, c, _, _ in steps_for(read_eviews_xml(XML))] == ["AddFactors", "SolveModel", "OptimalControlCommand"]


def test_the_forms_build_offscreen(frbus):
    pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import command_dialog

    for cls in (ShockDecomposition, ExportModel, ImportFRBUS, ImportEViewsStateSpace, AddFactors, OptimalControlCommand):
        assert command_dialog(cls, frbus).editors
    solve = command_dialog(SolveModel, frbus)
    assert solve.on_advanced_tab("residuals") and not solve.row_visible("residuals")
    solve.editors["errors"].set("sample")
    assert solve.row_visible("residuals") and solve.row_visible("residual_series")
    export = command_dialog(ExportModel, frbus)
    assert not export.row_visible("form")
    export.editors["syntax"].set("eviews")
    assert export.row_visible("form")
