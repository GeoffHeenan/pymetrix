"""Model > Analysis > Impulse responses: a solved linear (rational-expectations) model's responses to one shock.

The dialog runs ``LinearSolution.irf(periods, shock_sd={shock: size}, shocks=[shock])``, a recorded call. Checked
against closed forms: an AR(1), y(t) = rho y(t-1) + e(t), responds size * rho**h at horizon h; and any solution
x(t) = P x(t-1) + Q e(t) responds P**h Q[:, j] * size, computed here from the decision rules by hand. The dialog's
result equals the API call, the code typed by hand records the same call, and the script replays."""

import math
import os
import warnings

import numpy as np
import pytest

from pymetrix import ModelSource, templates
from pymetrix.commands import COMMANDS, LoadTemplate, ModelImpulseResponses, Session
from pymetrix.errors import SpecificationError
from pymetrix.forms import form_spec

AR1 = """
parameters
    rho = 0.8
shocks
    e = 0.5
variables
    y = 0
equations
    y = rho*y[-1] + e
"""


def _session():
    s = Session()
    s.objects["ar"] = ModelSource(AR1, "pymetrix", "ar").build()
    s.run_code("ar_sol = ar.solve()")
    s.run(LoadTemplate("new_keynesian", name="nk"))
    s.run_code("nk_sol = nk.solve()")
    return s


def _replay(s, name):
    ns = {"ar": s["ar"]}  # put in the session's objects directly, not by a step
    exec(s.script(), ns)
    return ns[name]


def test_an_ar1_responds_size_times_rho_to_the_h():
    s = _session()
    s.run(ModelImpulseResponses("ar_sol", "e", size=0.5, periods=10, name="ar_irf"))
    irf = s["ar_irf"]
    assert list(irf.shocks) == ["e"] and list(irf.responses) == ["y"] and irf.values.shape == (11, 1, 1)
    assert np.allclose(irf.values[:, 0, 0], [0.5 * 0.8 ** h for h in range(11)], rtol=1e-12, atol=1e-15)
    s.run(ModelImpulseResponses("ar_sol", "e", size=-2.0, periods=3, name="down"))
    assert np.allclose(s["down"].values[:, 0, 0], [-2.0, -1.6, -1.28, -1.024])  # linear: a negative shock mirrors


def test_a_forward_looking_model_responds_as_its_decision_rules_say():
    s = _session()
    s.run(ModelImpulseResponses("nk_sol", "ei", size=0.25, periods=12))
    sol, irf = s["nk_sol"], s["irf"]
    j, k = list(sol.shocks).index("ei"), len(sol.model.observed)
    by_hand = np.array([(np.linalg.matrix_power(sol.P, h) @ sol.Q[:, j])[:k] * 0.25 for h in range(13)])
    assert np.allclose(irf.values[:, :, 0], by_hand, rtol=1e-10, atol=1e-14)
    assert list(irf.shocks) == ["ei"] and list(irf.responses) == list(sol.model.observed)
    assert abs(irf.values[-1, :, 0]).max() < abs(irf.values[0, :, 0]).max()  # a stable solution's responses die out


def test_the_dialog_equals_the_api_call_and_its_script_replays():
    s = _session()
    dialog = ModelImpulseResponses("nk_sol", "ey", size=0.5, periods=8, name="demand")
    s.run(dialog)
    record = s.history[-1]
    assert record.ok, record.error
    assert record.code == "demand = nk_sol.irf(periods=8, shock_sd={'ey': 0.5}, shocks=['ey'])" == dialog.to_code()
    assert record.call.function == "pymetrix:linear_solution.irf" and record.call.to_code() == record.code
    by_api = templates.new_keynesian().solve().irf(8, shock_sd={"ey": 0.5}, shocks=["ey"])
    assert np.array_equal(s["demand"].values, by_api.values)
    typed = _session()
    typed.run_code(record.code)  # the code the dialog wrote, typed by hand: the same call
    assert type(typed.history[-1].command).__name__ == "Call" and typed.history[-1].call.same_as(record.call)
    assert np.array_equal(typed["demand"].values, s["demand"].values)
    assert np.array_equal(_replay(s, "demand").values, s["demand"].values)  # the script replays
    assert "nk_sol = nk.solve()" in s.script()


def test_the_shock_field_lists_the_solutions_shocks_and_bad_choices_are_refused():
    s = _session()
    spec = {f.name: f for f in form_spec(ModelImpulseResponses)}
    assert spec["shock"].dynamic and spec["shock"].depends_on == "solution"
    assert s.choices(spec["shock"], {"solution": "nk_sol"}) == ["ey", "epi", "ei"]
    assert s.choices(spec["shock"], {"solution": "ar_sol"}) == ["e"] and s.choices(spec["shock"], {}) == []
    assert s.choices(spec["solution"], {}) == ["ar_sol", "nk_sol"]  # only solved linear models are offered
    assert ModelImpulseResponses("nk_sol", "e").validate(s) == ["nk_sol has no shock 'e'; its shocks are ey, epi, ei"]
    assert ModelImpulseResponses("nk", "ey").validate(s)[0].startswith("nk is a Template, not a solved linear model")
    assert ModelImpulseResponses("gone", "ey").validate(s) == ["no solution named 'gone'"]
    assert ModelImpulseResponses("nk_sol", "ey", size=0.0).validate(s) == ["give the shock a size other than 0 (1 for one unit)"]
    assert ModelImpulseResponses("nk_sol", "ey", size=math.nan).validate(s)
    assert ModelImpulseResponses("nk_sol", "ey").validate(s) == []
    assert {f: COMMANDS["ModelImpulseResponses"].cls.__dataclass_fields__[f].default
            for f in ("size", "periods", "name")} == {"size": 1.0, "periods": 20, "name": "irf"}


def test_the_api_selects_shocks_without_changing_their_responses():
    sol = templates.new_keynesian().solve()
    every = sol.irf(6, shock_corr={("ey", "ei"): 0.3})  # correlated: orthogonalised over all the shocks
    picked = sol.irf(6, shock_corr={("ey", "ei"): 0.3}, shocks=["ei", "ey"])
    assert list(picked.shocks) == ["ei", "ey"]
    assert np.allclose(picked.values, every.values[:, :, [2, 0]])
    assert np.array_equal(sol.irf(6, shocks="epi").values, sol.irf(6).values[:, :, [1]])
    with pytest.raises(SpecificationError, match="no shock 'nope'; its shocks are ey, epi, ei"):
        sol.irf(6, shocks=["nope"])
    with pytest.raises(SpecificationError, match="give at least one shock"):
        sol.irf(6, shocks=[])
    with pytest.raises(SpecificationError, match="a whole number at least 0"):
        sol.irf(-1)


def test_the_menu_and_next_steps_offer_it():
    from pymetrix.commands.menus import menu_order
    from pymetrix.ui.next_steps import steps_for

    paths = {i.cls.__name__: p for i, p in menu_order([i for i in COMMANDS.values() if i.menu])}
    assert paths["ModelImpulseResponses"] == "Model → Analysis → Impulse responses…"
    from pymetrix.wording import dialog_title

    title = dialog_title(ModelImpulseResponses, COMMANDS["ModelImpulseResponses"].menu)
    assert title == "Impulse responses of a solved model"  # the VAR dialog is "Impulse responses": help pages need two names
    from pymetrix.help import contents

    pages = [page for _, entries in contents() for page, _ in entries]
    assert "Impulse responses of a solved model" in pages and "Impulse responses" in pages
    assert ("Impulse responses…", ModelImpulseResponses, "solution", "model") in steps_for(templates.rbc().solve())


def test_the_dialog_offscreen():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import command_dialog

    s = _session()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dialog = command_dialog(ModelImpulseResponses, s)
    try:
        assert [dialog.editors[f].spec.label for f in ("solution", "shock", "size", "periods", "name")][:4] == \
            ["Solution", "Shock", "Shock size", "Horizon"]
        dialog.editors["solution"].set("nk_sol")
        app.processEvents()
        shock = dialog.editors["shock"].widget
        assert [shock.itemText(i) for i in range(shock.count())] == ["ey", "epi", "ei"]  # follows the solution chosen
        dialog.editors["shock"].set("epi")
        dialog.editors["size"].set(0.75)
        dialog.editors["periods"].set(6)
        app.processEvents()
        shown = dialog.code_text()
        assert shown == "irf = nk_sol.irf(periods=6, shock_sd={'epi': 0.75}, shocks=['epi'])"
        ok = dialog.findChild(QtWidgets.QDialogButtonBox).button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        assert ok.isEnabled()
        ok.click()
        app.processEvents()
        assert dialog.command == ModelImpulseResponses("nk_sol", "epi", size=0.75, periods=6)
        assert s.history[-1].code == shown  # what Show the code showed is what was recorded
        assert np.array_equal(s["irf"].values, s["nk_sol"].irf(6, shock_sd={"epi": 0.75}, shocks=["epi"]).values)
        dialog.editors["solution"].set("ar_sol")
        app.processEvents()
        assert [shock.itemText(i) for i in range(shock.count())] == ["e"]
    finally:
        dialog.close()
