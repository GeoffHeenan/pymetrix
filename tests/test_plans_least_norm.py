"""Least-norm conditional forecasts on first-order solutions: more free shocks than holds, weights (free_sd) and soft
holds (hold_sd).

Checked against the same problem solved independently: the responses of the held variables to each free shock in
each period are taken from separate simulations (the model is linear), and the weighted minimum-norm solution, and the
soft one, from numpy's pseudo-inverse and least squares on the stacked problem."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, FirstOrder, PerfectForesight, Scenario, templates
from pymetrix.models import ModelError

SPAN = ("2025Q1", "2027Q4")
QUARTERS = [str(q) for q in pd.period_range("2025Q1", "2025Q4", freq="Q")]


@pytest.fixture(scope="module")
def sol():
    return templates.new_keynesian().solve()


def _responses(sol, free, held, last, unanticipated=()):
    """Rows: the held (variable, period) pairs; columns: each free shock in each period up to ``last``."""
    cols = []
    for s in free:
        for t in range(last + 1):
            path = sol.solve(None, SPAN, Scenario(set={s: {QUARTERS[t]: 1.0}}, unanticipated=list(unanticipated)))
            cols.append([path[v].loc[q] for v, q in held])
    return np.array(cols).T


def test_square_plans_reproduce_the_paired_solution(sol):
    hold = {"i": dict.fromkeys(QUARTERS, 0.5)}
    for unanticipated in ((), ("ei",)):
        paired = sol.solve(None, SPAN, Scenario(hold=hold, free=["ei"], unanticipated=list(unanticipated)))
        norm = sol.solve(None, SPAN, Scenario(hold=hold, free=["ei"], free_sd={"ei": 1.0}, unanticipated=list(unanticipated)))
        np.testing.assert_allclose(norm.frame.loc["2025Q1":, ["y", "pi", "i", "ei"]], paired.frame.loc["2025Q1":, ["y", "pi", "i", "ei"]],
                                   atol=1e-10)
        assert norm.provenance.method["conditioning"] == "least norm" and "conditioning" not in paired.provenance.method


def test_more_free_shocks_than_holds_is_the_weighted_minimum_norm(sol):
    held = [("i", "2025Q1"), ("i", "2025Q2")]
    A = _responses(sol, ["ei", "ey"], held, last=1)
    g = np.full(2, 0.5)
    plan = sol.solve(None, SPAN, Scenario(hold={"i": {"2025Q1": 0.5, "2025Q2": 0.5}}, free=["ei", "ey"]))
    np.testing.assert_allclose(plan["i"].loc["2025Q1":"2025Q2"], 0.5, atol=1e-12)  # exact holds are met
    expected = np.linalg.pinv(A) @ g  # the minimum-norm shocks: ei then ey, in 2025Q1 and 2025Q2
    np.testing.assert_allclose([plan["ei"].loc["2025Q1"], plan["ei"].loc["2025Q2"], plan["ey"].loc["2025Q1"], plan["ey"].loc["2025Q2"]],
                               expected, atol=1e-10)
    assert (plan["ei"].loc["2025Q3":] == 0).all() and (plan["ey"].loc["2025Q3":] == 0).all()  # free only up to the last hold
    # weights: shocks in units of free_sd, so a small free_sd makes a shock expensive
    w = np.repeat([1.0, 0.1], 2)
    weighted = sol.solve(None, SPAN, Scenario(hold={"i": {"2025Q1": 0.5, "2025Q2": 0.5}}, free=["ei", "ey"], free_sd={"ey": 0.1}))
    expected = w * (np.linalg.pinv(A * w) @ g)
    np.testing.assert_allclose([weighted["ei"].loc["2025Q1"], weighted["ei"].loc["2025Q2"], weighted["ey"].loc["2025Q1"],
                                weighted["ey"].loc["2025Q2"]], expected, atol=1e-10)
    assert np.abs(weighted["ey"].loc["2025Q1":"2025Q2"]).sum() < np.abs(plan["ey"].loc["2025Q1":"2025Q2"]).sum()


def test_soft_holds(sol):
    def plan(sd):
        return sol.solve(None, SPAN, Scenario(hold={"i": {"2025Q2": 1.0}}, free=["ei", "ey"], hold_sd={"i": sd}))

    exact = sol.solve(None, SPAN, Scenario(hold={"i": {"2025Q2": 1.0}}, free=["ei", "ey"]))
    assert exact["i"].loc["2025Q2"] == pytest.approx(1.0, abs=1e-12)
    for zero in (0, 0.0, None):
        np.testing.assert_array_equal(plan(zero).frame.loc["2025Q1":, ["i", "ei", "ey"]].to_numpy(),
                                      exact.frame.loc["2025Q1":, ["i", "ei", "ey"]].to_numpy())
    misses = [abs(plan(sd)["i"].loc["2025Q2"] - 1.0) for sd in (1e-6, 0.1, 1.0, 1e3)]
    assert misses[0] < 1e-8 and misses[0] < misses[1] < misses[2] < misses[3] and misses[3] > 0.99  # met more as the sd shrinks
    # against the stacked least-squares problem: min |u|^2 + ((A u - g) / sd)^2
    A = _responses(sol, ["ei", "ey"], [("i", "2025Q2")], last=1)
    sd = 0.3
    u = np.linalg.lstsq(np.vstack([np.eye(4), A / sd]), np.concatenate([np.zeros(4), [1.0 / sd]]), rcond=None)[0]
    soft = plan(sd)
    np.testing.assert_allclose([soft["ei"].loc["2025Q1"], soft["ei"].loc["2025Q2"], soft["ey"].loc["2025Q1"], soft["ey"].loc["2025Q2"]],
                               u, atol=1e-10)


def test_exact_and_soft_together(sol):
    hold = {"i": {"2025Q1": 0.5, "2025Q3": 0.2}, "y": {"2025Q2": -1.0}}
    plan = sol.solve(None, SPAN, Scenario(hold=hold, free=["ei", "ey", "epi"], hold_sd={"i": {"2025Q3": 0.05}}))
    assert plan["i"].loc["2025Q1"] == pytest.approx(0.5, abs=1e-10) and plan["y"].loc["2025Q2"] == pytest.approx(-1.0, abs=1e-10)
    assert 0 < abs(plan["i"].loc["2025Q3"] - 0.2) < 0.2  # soft: near, not on


def test_refusals(sol):
    with pytest.raises(ModelError, match="cannot meet the exact holds"):  # two exact holds, one surprise in one period
        sol.solve(None, SPAN, Scenario(hold={"i": {"2025Q1": 0.5}, "y": {"2025Q1": 0.1}}, free=["ei"], unanticipated=["ei"]))
    with pytest.raises(ModelError, match="negative"):
        sol.solve(None, SPAN, Scenario(hold={"i": {"2025Q1": 0.5}}, free=["ei", "ey"], hold_sd={"i": -1.0}))
    with pytest.raises(ModelError, match="positive"):
        sol.solve(None, SPAN, Scenario(hold={"i": {"2025Q1": 0.5}}, free=["ei"], free_sd={"ei": 0.0}))
    with pytest.raises(ModelError, match="held variables"):
        sol.solve(None, SPAN, Scenario(hold={"i": {"2025Q1": 0.5}}, free=["ei"], hold_sd={"y": 1.0}))
    with pytest.raises(ModelError, match="0 held variables but 1 free"):
        sol.solve(None, SPAN, Scenario(free=["ei"]))
    nk = templates.new_keynesian()
    idx = pd.period_range("2024Q1", "2030Q4", freq="Q", name="time")
    flat = Dataset.from_frame(pd.DataFrame({v: 0.0 for v in [*nk.model.endogenous, *nk.model.exogenous]}, index=idx))
    soft = Scenario(hold={"i": {"2025Q1": 0.5}}, free=["ei"], hold_sd={"i": 0.1})
    with pytest.raises(ModelError, match="first-order solution"):
        nk.model.solve(flat, SPAN, soft, solver=PerfectForesight(terminal="steady_state"))
    via_first_order = nk.model.solve(flat, SPAN, soft, solver=FirstOrder())
    np.testing.assert_allclose(via_first_order["i"].loc["2025Q1":"2027Q4"], sol.solve(None, SPAN, soft)["i"].loc["2025Q1":"2027Q4"], atol=1e-10)
