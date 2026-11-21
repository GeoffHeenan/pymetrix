"""The compiled model: one indexed representation, parameters with status, and checks before solving."""

import numpy as np
import pytest

from pymetrix import Dataset, Equation, Model, OLS, Parameter, ParameterSet
from pymetrix.metadata import Param
from pymetrix.models import ModelError


@pytest.fixture(scope="module")
def data():
    return Dataset.example("sample_macro")


@pytest.fixture(scope="module")
def lab2(data):
    cons = Equation("c ~ y + c[-1]", name="cons").fit(data, method=OLS(cov="hac"))
    inv = Equation("i ~ y[-1]", name="inv").fit(data)
    return Model([cons, inv, Equation("y = c + i + g")]), cons


def _nk(phi=1.5):
    eqs = [Equation("y ~ y[+1] - s*(i - pi[+1]) + ey", params=["s"]), Equation("pi ~ b*pi[+1] + k*y + epi", params=["b", "k"]),
           Equation("i ~ r*i[-1] + (1-r)*(phi*pi + 0.5*y) + ei", params=["r", "phi"])]
    return Model(eqs, params={"s": 1, "b": 0.99, "k": 0.1, "r": 0.8, "phi": phi})


def test_compile_indexes_the_model(lab2):
    model, cons = lab2
    system = model.compile()
    assert system.endogenous == ("c", "i", "y") and system.exogenous == ("g",) and system.shocks == ()
    assert system.incidence == {"c": (-1, 0), "i": (0,), "y": (-1, 0), "g": (0,)}
    assert (system.max_lag, system.max_lead) == (1, 0)
    params = system.parameters
    assert params.names == ("cons.const", "cons.y", "cons.c[-1]", "inv.const", "inv.y[-1]")  # named by equation
    assert params.by_status("estimated") == params.names
    assert params["cons.y"].value == pytest.approx(cons.params["y"]) and params["cons.y"].se == pytest.approx(cons.params.se[1])
    assert params["cons.y"].equation == "cons"


def test_every_solution_path_uses_the_compiled_model(lab2, data):
    model, _ = lab2
    system = model.compile()
    resolve = lambda name, lag: float(data[name].iloc[50 + lag])  # noqa: E731
    from pymetrix.expr import evaluate

    for compiled, used in zip(system.residuals(), model.model_equations):
        assert evaluate(compiled, resolve) == pytest.approx(evaluate(used.residual, resolve))
    shifted = system.residuals({"cons.y": 0.0})  # other values without rebuilding the model
    assert evaluate(shifted[0], resolve) != pytest.approx(evaluate(model.model_equations[0].residual, resolve))
    assert model.parameters is system.parameters


def test_parameter_sets():
    ps = ParameterSet.from_values({"b": 0.99, "k": 0.1})
    assert ps.values() == {"b": 0.99, "k": 0.1} and ps.by_status("calibrated") == ("b", "k")
    fixed = ps.fix("b")
    assert fixed["b"].status == "fixed" and ps["b"].status == "calibrated"  # values, not mutated
    assert fixed.with_values({"k": 0.2})["k"].value == 0.2
    with pytest.raises(KeyError, match="no parameters"):
        ps.with_values({"z": 1})
    with pytest.raises(ValueError, match="unique"):
        ParameterSet((Parameter("a", 1.0), Parameter("a", 2.0)))
    assert "fixed" in str(fixed.table())


def test_calibrated_parameters_are_shared_and_carry_metadata():
    eqs = [Equation("x ~ rho*x[-1] + u", params=[Param("rho", bounds=(0.0, 1.0))]), Equation("z ~ rho*z[-1] + x", params=["rho"])]
    model = Model(eqs, params=ParameterSet.from_values({"rho": 0.5}))
    assert model.parameters.names == ("rho",) and model.parameters["rho"].bounds == (0.0, 1.0)  # one parameter, in two equations
    assert model.check().ok
    outside = Model(eqs, params={"rho": 1.5}).check()
    assert dict((c, r) for c, r, _ in outside.rows)["parameters"] == "warning"


def test_checks_pass_and_fail(lab2, data):
    model, _ = lab2
    history = model.check(data, ("2015Q1", "2024Q4"))
    assert history.ok and all(r in ("ok", "note") for _, r, _ in history.rows)
    future = dict((c, (r, d)) for c, r, d in model.check(data, ("2025Q1", "2026Q4")).rows)
    assert future["exogenous paths"][0] == "warning" and "Scenario" in future["exogenous paths"][1]  # a scenario can set g
    first = model.check(data, ("2000Q1", "2004Q4"))
    assert not first.ok and any("c, y" in e or "starting" in e for e in first.errors())  # c[-1] and y[-1] before the data
    missing = Model([Equation("y = c + zz")]).check(data)
    assert not missing.ok and "zz" in missing.errors()[0]
    circular = Model([Equation("y = y + c")]).check()
    assert "targets" in circular.errors()[0]


def test_forward_looking_checks():
    good = _nk().check(shocks=["ey", "epi", "ei"])
    rows = dict((c, r) for c, r, _ in good.rows)
    assert good.ok and rows["steady state"] == "ok" and rows["Blanchard-Kahn"] == "ok" and rows["timing"] == "note"
    passive = _nk(phi=0.5).check(shocks=["ey", "epi", "ei"])  # a Taylor rule that does not satisfy the Taylor principle
    assert not passive.ok and "Blanchard-Kahn" in passive.errors()[0] and "not unique" in passive.errors()[0]
    unknown = _nk().check(shocks=["ey", "q"])
    assert any("q is not exogenous" in e for e in unknown.errors())
    lagged = Model([Equation("x ~ a*x[+1] + u[-1]", params=["a"])], params={"a": 0.5}).check(shocks=["u"])
    assert any("enters with a lag or lead" in e for e in lagged.errors())


def test_the_check_dialog_views_and_project(lab2, data, tmp_path):
    from pymetrix.commands import CheckModel, DefineEquation, FitEquation, Session
    from pymetrix.testing import check_project
    from pymetrix.views import view_sections

    s = Session()
    s.objects["data"] = data
    s.run(FitEquation("c ~ y + c[-1]", name="cons"))
    s.run(FitEquation("i ~ y[-1]", name="inv"))
    s.run(DefineEquation("y = c + i + g", name="identity"))
    check_project(s, CheckModel("cons, inv, identity", start="2015Q1", end="2024Q4"), tmp_path)
    assert "model_check = Model([cons, inv, identity]).check(data, span=('2015Q1', '2024Q4'))" in s.script()
    assert s["model_check"].ok
    assert [t for t, _ in view_sections(s["model_check"])] == ["Checks", "Equations", "Variables", "Parameters"]
    assert [t for t, _ in view_sections(lab2[0].compile())] == ["Equations", "Variables", "Parameters", "Incidence"]


def test_errors_are_unchanged():
    with pytest.raises(ModelError, match="needs values for b"):
        Model([Equation("pi ~ b*pi[+1] + y", params=["b"])])
    with pytest.raises(ModelError, match="more than one equation"):
        Model([Equation("y = c"), Equation("y = i")])
