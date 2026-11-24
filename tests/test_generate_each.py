"""Generate series for each: the For each field of Data → Generate series, Dataset.with_each and the expansion behind
them (pymetrix.data.each).

Checks are against values computed directly with numpy and against the explicit with_series / derive call each
expansion stands for; the step's script is that call and replays to the same data."""

import json
import os
import zipfile

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset
from pymetrix.commands import GenerateSeries, Session, load_project, save_project
from pymetrix.commands.registry import from_json, to_json, upgrade
from pymetrix.data.each import expand_each, fill_template, preview_each, split_series
from pymetrix.errors import PyMetrixError, SpecificationError
from pymetrix.selection import Selection


@pytest.fixture
def wide():
    index = pd.period_range("2001Q1", periods=16, freq="Q")
    rng = np.random.default_rng(7)
    cols = {n: 100 * np.exp(np.cumsum(rng.normal(0.01, 0.02, 16))) for n in ("gdp", "cons", "inv", "ip_us", "ip_de", "cpi", "u_sa_m", "x_sa_q")}
    return Dataset.from_frame(pd.DataFrame(cols, index=index))


def _session(data, name="data"):
    s = Session()
    s.objects[name] = data
    return s


def _replay(s, name="data", **objects):
    ns = dict(objects)
    exec(s.script(), ns)
    return ns[name]


# -- the expansion --------------------------------------------------------------------------------------------------


def test_a_list_expands_to_one_formula_per_series(wide):
    assert split_series("gdp cons,inv ,  cpi") == ["gdp", "cons", "inv", "cpi"]
    assert expand_each("gdp cons inv", "{x}_g", "100*d(log({x}))", wide) == {
        "gdp_g": "100*d(log(gdp))", "cons_g": "100*d(log(cons))", "inv_g": "100*d(log(inv))"}
    assert expand_each(["gdp"], "dl_{x}", "{x} / cpi * 100", wide) == {"dl_gdp": "gdp / cpi * 100"}
    assert expand_each(Selection(("cons", "gdp")), "r{s}", "{s}/cpi", wide) == {"rcons": "cons/cpi", "rgdp": "gdp/cpi"}  # {s} too


def test_wildcards_give_their_parts_as_numbered_placeholders(wide):
    assert expand_each("ip_*", "{1}_growth", "100*dlog({x})", wide) == {"us_growth": "100*dlog(ip_us)", "de_growth": "100*dlog(ip_de)"}
    assert expand_each("*_sa_?", "{1}_{2}", "{x}", wide) == {"u_m": "u_sa_m", "x_q": "x_sa_q"}
    assert expand_each("*_*", "{2}{1}", "{x}", ["a_b_c"]) == {"ca_b": "a_b_c"}  # each * as much as it can, left first
    assert expand_each("gdp ip_*", "{x}_g", "{x}", wide) == {"gdp_g": "gdp", "ip_us_g": "ip_us", "ip_de_g": "ip_de"}  # mixed
    assert list(expand_each("ip_* ip_us", "{x}2", "{x}", wide)) == ["ip_us2", "ip_de2"]  # each series once, in dataset order per entry


def test_placeholders_and_escaping():
    assert fill_template("{{lit}} {x} {1}", "ip_us", ("us",)) == "{lit} ip_us us"
    for template, match in (("{y}", "not a placeholder"), ("{2}", "matched 1 part"), ("{x", "lone"), ("x}", "lone")):
        with pytest.raises(SpecificationError, match=match):
            fill_template(template, "ip_us", ("us",))
    with pytest.raises(SpecificationError, match="not matched by a wildcard"):
        fill_template("{1}", "gdp")


def test_names_formulas_and_matches_are_checked(wide):
    found = preview_each(wide, "gdp cons", "{x}-g", "log({x}")
    assert not found.ok and [r.status for r in found.rows] == ["error", "error"]
    assert "not a valid series name" in found.rows[0].note
    found = preview_each(wide, "gdp", "{x}_l", "log({x}")
    assert "does not parse" in found.rows[0].note and "expected ')'" in found.rows[0].note
    assert "no series nope" in preview_each(wide, "gdp", "{x}_r", "{x}/nope").rows[0].note
    assert "Python keyword" in preview_each(wide, "gdp", "lambda", "{x}").rows[0].note
    assert preview_each(wide, "zz_*", "{x}_g", "{x}").errors() == ["no series match zz_*"]  # empty match: said plainly
    assert preview_each(wide, "gdpx", "{x}_g", "{x}").errors() == ["no series gdpx"]
    assert preview_each(wide, "", "{x}_g", "{x}").errors() == ["nothing to generate: For each names no series"]
    assert "every series would get the same name" in preview_each(wide, "gdp cons", "g", "{x}").errors()[0]
    listed = preview_each(wide, "ip_us ip_de", "{1}_g", "{x}")  # listed names have no wildcard parts
    assert listed.rows[0].note == "{1} has nothing for ip_us: it was not matched by a wildcard"
    twice = preview_each(wide, "u_sa_m x_sa_q", "{x}", "{x}", replace=True)
    assert twice.ok  # replacing themselves is allowed with Replace
    dup = preview_each(["ip_us", "xp_us"], "?p_*", "{2}", "{x}")
    assert [r.status for r in dup.rows] == ["new", "error"] and dup.rows[1].note.startswith("made twice (also from ip_us)")
    with pytest.raises(SpecificationError, match="no series match"):
        expand_each("zz_*", "{x}_g", "{x}", wide)


def test_clashes_and_replace(wide):
    found = preview_each(wide, "gdp cons", "{x}", "{x} * 2")
    assert [r.status for r in found.rows] == ["exists", "exists"]
    assert found.errors() == ["data already has series gdp, cons: choose another name, or tick Replace (replace=True) to overwrite"]
    found = preview_each(wide, "gdp cons", "{x}", "{x} * 2", replace=True)
    assert found.ok and [r.status for r in found.rows] == ["replaces", "replaces"]
    with pytest.raises(SpecificationError, match="tick Replace"):
        wide.with_each("gdp", "{x}", "{x} * 2")
    doubled = wide.with_each("gdp cons", "{x}", "{x} * 2", replace=True)
    np.testing.assert_allclose(doubled["gdp"].to_numpy(), 2 * wide["gdp"].to_numpy())


# -- Dataset.with_each -------------------------------------------------------------------------------------------------


def test_with_each_equals_explicit_with_series_and_numpy(wide):
    made = wide.with_each("gdp cons inv", "{x}_g", "100*d(log({x}))")
    explicit = wide.with_series(gdp_g="100*d(log(gdp))", cons_g="100*d(log(cons))", inv_g="100*d(log(inv))")
    pd.testing.assert_frame_equal(made.frame, explicit.frame)
    for n in ("gdp", "cons", "inv"):  # independent: numpy's log differences
        x = wide[n].to_numpy()
        np.testing.assert_allclose(made[f"{n}_g"].to_numpy()[1:], 100 * np.diff(np.log(x)))
        assert np.isnan(made[f"{n}_g"].to_numpy()[0])
    assert made.derived == {}  # values, computed once


def test_with_each_derived_follows_its_inputs(wide):
    made = wide.with_each("ip_*", "{1}_r", "{x} / cpi * 100", derived=True)
    assert set(made.derived) == {"us_r", "de_r"}
    pd.testing.assert_frame_equal(made.frame, wide.derive(us_r="ip_us / cpi * 100", de_r="ip_de / cpi * 100").frame)
    moved = made.with_series(cpi="cpi * 2")
    np.testing.assert_allclose(moved["us_r"].to_numpy(), made["us_r"].to_numpy() / 2)
    chained = wide.with_each("gdp", "{x}_l", "log({x})", derived=True)  # derived formulas may use the batch's own series
    assert "gdp_l" in chained.derived


def test_panel_datasets_work_per_entity():
    panel = Dataset.example("grunfeld")
    made = panel.with_each("invest value", "d{x}", "d({x})")
    for firm, block in panel.frame.groupby(level=0, sort=False):
        got = made.frame.loc[firm, "dinvest"].to_numpy()
        assert np.isnan(got[0])  # no difference across firms
        np.testing.assert_allclose(got[1:], np.diff(block["invest"].to_numpy()))
    s = _session(panel)
    s.run(GenerateSeries("d{x}", "d({x})", for_each="invest value"))
    pd.testing.assert_frame_equal(s["data"].frame, made.frame)
    pd.testing.assert_frame_equal(_replay(s, data=panel).frame, made.frame)


# -- the dialog's step ------------------------------------------------------------------------------------------------


def test_the_step_records_the_expansion_as_one_call_that_replays(wide):
    s = _session(wide)
    s.run(GenerateSeries("{x}_g", "100*d(log({x}))", for_each="gdp cons inv"))
    code = "data = data.with_series(gdp_g='100*d(log(gdp))', cons_g='100*d(log(cons))', inv_g='100*d(log(inv))')"
    assert s.history[-1].code == code and code in s.script()
    explicit = wide.with_series(gdp_g="100*d(log(gdp))", cons_g="100*d(log(cons))", inv_g="100*d(log(inv))")
    pd.testing.assert_frame_equal(s["data"].frame, explicit.frame)
    pd.testing.assert_frame_equal(_replay(s, data=wide).frame, explicit.frame)
    assert s["data"].frame.equals(wide.with_each("gdp cons inv", "{x}_g", "100*d(log({x}))").frame)  # = the API helper
    call = s.history[-1].call
    assert call.function == "pymetrix:dataset.with_series" and list(call.arguments["exprs"]) == ["gdp_g", "cons_g", "inv_g"]


def test_keep_updated_records_derive_and_a_wildcard_is_matched_before_the_step(wide):
    s = _session(wide)
    s.run(GenerateSeries("l{x}", "log({x})", for_each="*", keep_updated=True))
    names = list(wide.names)
    assert s.history[-1].code == "data = data.derive(" + ", ".join(f"l{n}='log({n})'" for n in names) + ")"
    assert list(s.history[-1].call.arguments["recipes"]) == [f"l{n}" for n in names]  # not the series the step made
    assert set(s["data"].derived) == {f"l{n}" for n in names}
    pd.testing.assert_frame_equal(_replay(s, data=wide).frame, s["data"].frame)


def test_the_single_case_is_unchanged(wide):
    s = _session(wide)
    s.run(GenerateSeries("z", "gdp * 2"))
    assert s.history[-1].code == "data = data.with_series(z='gdp * 2')"
    assert GenerateSeries("z", "gdp * 2").to_code() == "data = data.with_series(z='gdp * 2')"
    with pytest.raises(PyMetrixError, match="not a valid series name"):
        s.run(GenerateSeries("{x}_g", "gdp"))  # a template without For each is a name, and not a valid one
    assert GenerateSeries("z", "y").preview_rows(s) is None  # no preview for one series


def test_validation_messages(wide):
    s = _session(wide)
    assert GenerateSeries("{x}_g", "{x}", for_each="zz_*").validate(s) == ["no series match zz_*"]
    assert GenerateSeries("{x}", "{x}", for_each="gdp").validate(s)[0].startswith("data already has a series gdp")
    assert GenerateSeries("{x}", "{x}", for_each="gdp", replace=True).validate(s) == []
    assert GenerateSeries("{x}_g", "{x}", data="nope", for_each="gdp").validate(s) == ["no dataset named 'nope'"]
    with pytest.raises(PyMetrixError, match="no series match zz_"):
        s.run(GenerateSeries("{x}_g", "{x}", for_each="zz_*"))
    assert not s.history and "zz_" not in s.script()  # refused before it runs: nothing recorded


def test_to_code_without_a_session_is_the_helper_and_runs(wide):
    cmd = GenerateSeries("{1}_g", "100*dlog({x})", for_each="ip_*", keep_updated=True, replace=True)
    code = cmd.to_code()
    assert code == "data = data.with_each('ip_*', '{1}_g', '100*dlog({x})', derived=True, replace=True)"
    ns = {"data": wide}
    exec(code, ns)
    pd.testing.assert_frame_equal(ns["data"].frame, wide.with_each("ip_*", "{1}_g", "100*dlog({x})", derived=True).frame)


def test_typed_with_each_is_recorded_as_the_expanded_call(wide):
    s = _session(wide)
    s.run_code('data = data.with_each("gdp cons", "{x}_g", "100*d(log({x}))")')
    assert "data = data.with_series(gdp_g='100*d(log(gdp))', cons_g='100*d(log(cons))')" in s.script()
    assert "with_each" not in s.script()
    pd.testing.assert_frame_equal(_replay(s, data=wide).frame, s["data"].frame)


def test_undo_redo(wide):
    s = _session(wide)
    s.run(GenerateSeries("{x}_g", "100*d(log({x}))", for_each="gdp cons"))
    after = s["data"].frame.copy()
    s.undo()
    assert s["data"].names == wide.names and not s.history
    s.redo()
    pd.testing.assert_frame_equal(s["data"].frame, after)
    assert s.history[-1].command.for_each == "gdp cons"


def test_project_round_trip_keeps_the_patterns_and_the_call(wide, tmp_path):
    s = _session(wide)
    s.run(GenerateSeries("{1}_growth", "100*dlog({x})", for_each="ip_*"))
    path = save_project(s, tmp_path / "each.pymx")
    with zipfile.ZipFile(path) as z:
        step = json.loads(z.read("commands.json"))
    step = next(e for e in (step if isinstance(step, list) else step["steps"]) if e.get("command") == "GenerateSeries")
    assert step["fields"]["for_each"] == "ip_*" and step["fields"]["series"] == "{1}_growth"  # the patterns, to reopen
    assert step["call"]["arguments"]["exprs"] == {"us_growth": "100*dlog(ip_us)", "de_growth": "100*dlog(ip_de)"}  # the expansion
    opened = load_project(path)
    assert opened.script() == s.script()
    pd.testing.assert_frame_equal(opened["data"].frame, s["data"].frame)
    assert opened.history[-1].command == s.history[-1].command


def test_command_contract_schema_and_old_steps(wide):
    assert GenerateSeries.schema_version == 3  # 3: within, start and end (test_assign_within.py); for_each came with a default
    old = {"series": "z", "formula": "gdp*2", "data": "data", "keep_updated": False, "replace": False}  # saved before
    assert upgrade("pymetrix:generate_series", 2, dict(old))["series"] == "z"
    cmd = from_json({"kind": "command", "id": "pymetrix:generate_series", "version": 2, "fields": old})
    assert cmd.for_each == "" and cmd.to_code() == "data = data.with_series(z='gdp*2')"
    assert from_json({"kind": "command", "id": "pymetrix:generate_series", "version": 1,
                      "fields": {"series": "z", "formula": "gdp"}}).replace  # schema 1 still migrates
    round_trip = GenerateSeries("{x}_g", "{x}", for_each="gdp cons")
    assert from_json(json.loads(json.dumps(to_json(round_trip)))) == round_trip
    assert "target" not in {f for f in GenerateSeries.__dataclass_fields__}


# -- the dialog -------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_the_dialog_previews_every_row(app, wide):
    from pymetrix.ui.qt import command_dialog

    s = _session(wide)
    dialog = command_dialog(GenerateSeries, s, initial=GenerateSeries("{x}_g", "100*d(log({x}))", for_each="gdp ip_*"))
    dialog.refresh_rows_preview()
    table = dialog.rows_preview
    rows = [tuple(table.item(r, c).text() for c in range(3)) for r in range(table.rowCount())]
    assert rows == [("gdp_g", "100*d(log(gdp))", ""), ("ip_us_g", "100*d(log(ip_us))", ""), ("ip_de_g", "100*d(log(ip_de))", "")]
    assert dialog.rows_note.text() == "Preview: 3 new series"
    ok = dialog.findChild(__import__("PySide6.QtWidgets", fromlist=["QDialogButtonBox"]).QDialogButtonBox).button(
        __import__("PySide6.QtWidgets", fromlist=["QDialogButtonBox"]).QDialogButtonBox.StandardButton.Ok)
    assert ok.isEnabled()
    dialog.editors["series"].set("{x}")  # clashes: flagged, OK held until Replace
    dialog.refresh_rows_preview()
    assert table.item(0, 2).text() == "already exists: tick Replace to overwrite" and not ok.isEnabled()
    dialog.editors["replace"].set(True)
    dialog.refresh_rows_preview()
    assert table.item(0, 2).text() == "replaces the existing series" and ok.isEnabled()
    assert "3 replacing existing ones" in dialog.rows_note.text()
    dialog.editors["for_each"].set("zz_*")
    dialog.refresh_rows_preview()
    assert dialog.rows_note.text() == "no series match zz_*" and not ok.isEnabled() and table.isHidden()
    dialog.editors["for_each"].set("")  # one series: no preview, OK as before
    dialog.editors["series"].set("z")
    dialog.refresh_rows_preview()
    assert dialog.rows_note.isHidden() and ok.isEnabled()


def test_the_dialog_runs_the_expansion(app, wide):
    from pymetrix.ui.qt import command_dialog

    s = _session(wide)
    dialog = command_dialog(GenerateSeries, s)
    for field, value in (("series", "{1}_growth"), ("formula", "100*dlog({x})"), ("for_each", "ip_*")):
        dialog.editors[field].set(value)
    dialog.refresh_rows_preview()
    assert dialog.rows_preview.rowCount() == 2
    from PySide6.QtWidgets import QDialogButtonBox

    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()
    assert dialog.command == GenerateSeries("{1}_growth", "100*dlog({x})", for_each="ip_*")
    assert s.history[-1].code == "data = data.with_series(us_growth='100*dlog(ip_us)', de_growth='100*dlog(ip_de)')"
    pd.testing.assert_frame_equal(s["data"].frame, wide.with_each("ip_*", "{1}_growth", "100*dlog({x})").frame)


def test_the_series_browser_opens_it_with_the_series_picked(app, wide, monkeypatch):
    from PySide6.QtCore import QItemSelectionModel

    from pymetrix.ui import qt
    from pymetrix.ui.qt import main_window

    s = _session(wide)
    window = main_window(s)
    try:
        opened = []
        monkeypatch.setattr(qt, "command_dialog", lambda cls, session, parent=None, initial=None: opened.append(initial) or
                            type("Shut", (), {"exec": lambda self: False})())
        entries = window.palette_entries()
        from pymetrix.ui.palette import rank

        rank(entries, "data")[0].run()
        browser = window.mdi.subWindowList()[-1].widget().browser
        for row in (0, 1):
            browser.view.selectionModel().select(browser.proxy.index(row, 0), QItemSelectionModel.SelectionFlag.Select
                                                 | QItemSelectionModel.SelectionFlag.Rows)
        picked = browser.picked()
        browser.buttons["generate"].click()
        assert type(opened[-1]) is GenerateSeries and opened[-1].for_each == " ".join(picked) and opened[-1].data == "data"
        assert browser.buttons["generate"].text() == "Generate for each…"
    finally:
        s.dirty = False
        window.close()
