"""max(a, b) and min(a, b) in the model language: period by period, with derivatives, in data and in models."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, FirstOrder, Model, PerfectForesight, Scenario
from pymetrix.expr import ExpressionError, parse_expression
from pymetrix.expr.diff import diff
from pymetrix.expr.evaluate import evaluate
from pymetrix.expr.nodes import Name
from pymetrix.metadata import Role


def test_parsing_and_printing():
    assert str(parse_expression("min(a, b, 0)")) == "min(min(a, b), 0)"  # pairs, left to right
    for text in ("max(a, b)", "max(y)", "max(y, by=country)", "max(i_rule, 0.25) + step(x)"):
        assert str(parse_expression(str(parse_expression(text)))) == str(parse_expression(text))  # printing round-trips
    assert str(parse_expression("mean(y, by=country)")) == str(parse_expression("mean(y, country)"))
    assert parse_expression("max(a, b)").elementwise and not parse_expression("max(y, by=country)").elementwise
    assert {n.id for n in parse_expression("max(a, b)").refs()} == {"a", "b"}
    for bad, message in (("max(a, b, by=c)", "period by period"), ("log(x, by=c)", "does not take by="),
                         ("max(y, by=x[-1])", "by= takes an axis or a series name")):
        with pytest.raises(ExpressionError, match=message):
            parse_expression(bad)


def test_values_and_derivatives():
    e = parse_expression("max(2*a, b) - min(a, b^2)")
    f = lambda a, b: max(2 * a, b) - min(a, b ** 2)  # noqa: E731
    for a, b in ((1.0, 3.0), (2.0, 1.0), (0.5, 0.6)):  # each branch of each kink
        values = {"a": a, "b": b}
        resolve = lambda n, lag, v=values: v[n]  # noqa: E731
        assert evaluate(e, resolve) == pytest.approx(f(a, b))
        h = 1e-6
        assert evaluate(diff(e, Name("a")), resolve) == pytest.approx((f(a + h, b) - f(a - h, b)) / (2 * h), abs=1e-6)
        assert evaluate(diff(e, Name("b")), resolve) == pytest.approx((f(a, b + h) - f(a, b - h)) / (2 * h), abs=1e-6)
    s = pd.Series([-1.0, 0.5, np.nan])
    assert evaluate(parse_expression("max(x, 0)"), lambda n, lag: s).tolist()[:2] == [0.0, 0.5]
    assert np.isnan(evaluate(parse_expression("max(x, 0)"), lambda n, lag: s).iloc[2])  # missing stays missing


def test_in_data_and_panels():
    data = Dataset.example("sample_macro")
    assert np.allclose(data.eval("max(c - 900, 0)"), np.maximum(data["c"] - 900, 0))
    assert np.allclose(data.eval("max(c)"), data["c"].max())  # one argument: still the statistic
    rows = [{"country": c, "t": p, "y": float(10 * j + i)} for j, c in enumerate(["AUS", "NZL"]) for i, p in enumerate(pd.period_range("2018", periods=3, freq="Y"))]
    panel = Dataset.from_frame(pd.DataFrame(rows), time="t", entity="country")
    assert panel.eval("max(y, by=country)").tolist() == [2.0, 2.0, 2.0, 12.0, 12.0, 12.0]


def _nk(floor):
    eqs = [Equation("y ~ y[+1] - (1/sigma)*(i - pi[+1]) + ey", params=["sigma"]),
           Equation("pi ~ beta*pi[+1] + kappa*y + epi", params=["beta", "kappa"]),
           Equation("iu ~ rho*iu[-1] + (1-rho)*(phi_pi*pi + phi_y*y) + ei", params=["rho", "phi_pi", "phi_y"]),
           Equation("i ~ max(iu, floor) + wedge", params=["floor"])]
    return Model(eqs, params=dict(sigma=1, beta=0.99, kappa=0.1, rho=0.8, phi_pi=1.5, phi_y=0.5, floor=floor),
                 roles={"ey": Role.SHOCK, "epi": Role.SHOCK, "ei": Role.SHOCK, "wedge": Role.SHOCK})


def test_an_effective_lower_bound_under_perfect_foresight():
    model = _nk(-0.25)
    idx = pd.period_range("2024Q1", "2054Q4", freq="Q", name="time")
    flat = Dataset.from_frame(pd.DataFrame({v: 0.0 for v in [*model.endogenous, *model.exogenous]}, index=idx))
    demand = {"ey": {"2025Q1": -3.0, "2025Q2": -2.0, "2025Q3": -1.0}}
    pf = model.solve(flat, ("2025Q1", "2054Q4"), Scenario(set=demand), solver=PerfectForesight(terminal="steady_state"))
    i, iu = pf["i"].loc["2025Q1":"2034Q4"], pf["iu"].loc["2025Q1":"2034Q4"]
    np.testing.assert_allclose(i, np.maximum(iu, -0.25), atol=1e-10)
    binding = list(i.index[(iu < -0.25).to_numpy()].astype(str))
    assert binding == ["2025Q1", "2025Q2", "2025Q3"]
    # independently: the same model with the rate held at the floor in those quarters by an anticipated wedge (the
    # piecewise-linear method), solved by the first-order plan; the bound is slack at the steady state
    held = model.linearize().solve().solve(None, ("2025Q1", "2054Q4"),
                                            Scenario(set=demand, hold={"i": {q: -0.25 for q in binding}}, free=["wedge"]))
    for v in ("y", "pi", "i", "iu"):
        np.testing.assert_allclose(held[v].loc["2025Q1":"2034Q4"], pf[v].loc["2025Q1":"2034Q4"], atol=1e-8)
    free = model.solve(flat, ("2025Q1", "2054Q4"), Scenario(set=demand), solver=FirstOrder())
    assert free["i"].loc["2025Q1"] < -0.25 and pf["y"].loc["2025Q1"] < free["y"].loc["2025Q1"]  # the bound deepens the slump


def test_dynare_files_with_max():
    from pymetrix import read_mod

    dyn = read_mod("""
    var y i; varexo e; parameters rho floor top;
    rho = 0.5; floor = max(-1, -2); top = min(3, 4);
    model;
    y = rho*y(-1) + e;
    i = max(y, floor);
    end;
    """)
    assert dyn.params["floor"] == -1.0 and dyn.params["top"] == 3.0  # parameter values may use max and min too
    sol = dyn.solve()  # the bound is slack at the steady state (y = 0 > -1): i moves one for one with y
    assert sol.irf(2).values[0, list(sol.model.observed).index("i"), 0] == pytest.approx(1.0)
