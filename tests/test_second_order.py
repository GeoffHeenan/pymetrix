"""Second-order perturbation: checked against the exact solution of the growth model with log utility and full
depreciation (every second-order term known, no correction for risk), and against Dynare's own stoch_simul(order=2)
for a growth model with persistent technology (tests/fixtures/dynare-rbc-order2.json, from the dynare-reference
workflow; that test is skipped until the file is there)."""

import json
from pathlib import Path

import numpy as np
import pytest

from pymetrix import Equation, Model, read_mod, second_order, templates
from pymetrix.metadata import Role

FIXTURES = Path(__file__).parent / "fixtures"
ALPHA, BETA = 0.33, 0.96


@pytest.fixture(scope="module")
def exact_model():
    """k(t) = alpha beta exp(e) k(t-1)^alpha exactly; c = (1 - alpha beta) y."""
    m = Model([Equation("1/c ~ beta*alpha*y[+1]/(k*c[+1])", params=["alpha", "beta"]),
               Equation("y ~ exp(e)*k[-1]^alpha", params=["alpha"]),
               Equation("k = y - c")], params={"alpha": ALPHA, "beta": BETA}, roles={"e": Role.SHOCK})
    kb = (ALPHA * BETA) ** (1 / (1 - ALPHA))
    return m, kb


def test_every_term_equals_the_exact_solution(exact_model):
    m, kb = exact_model
    yb = kb**ALPHA
    s = second_order(m, ["e"], {"e": 0.02}, guess={"k": kb, "y": yb, "c": (1 - ALPHA * BETA) * yb})
    assert s.states == ("k",) and s.shocks == ("e",)
    k, y, c = (s.variables.index(v) for v in ("k", "y", "c"))
    # the derivatives of alpha beta exp(e) k^alpha at the steady state
    assert s.ghx[k, 0] == pytest.approx(ALPHA**2 * BETA * kb ** (ALPHA - 1), rel=1e-12)
    assert s.ghxx[k, 0] == pytest.approx(ALPHA**2 * BETA * (ALPHA - 1) * kb ** (ALPHA - 2), rel=1e-12)
    assert s.ghxu[k, 0] == pytest.approx(ALPHA**2 * BETA * kb ** (ALPHA - 1), rel=1e-12)
    assert s.ghuu[k, 0] == pytest.approx(kb, rel=1e-12)
    assert s.ghxx[y, 0] == pytest.approx(ALPHA * (ALPHA - 1) * kb ** (ALPHA - 2), rel=1e-12)
    assert s.ghxx[c, 0] == pytest.approx((1 - ALPHA * BETA) * ALPHA * (ALPHA - 1) * kb ** (ALPHA - 2), rel=1e-12)
    assert np.allclose(s.ghs2, 0.0, atol=1e-14)  # the policy does not depend on uncertainty: no correction for risk
    path = s.simulate(5, shocks=np.array([[0.05], [0.0], [0.0], [-0.03], [0.0]]))
    exact, kt = [], kb
    for e in (0.05, 0.0, 0.0, -0.03, 0.0):
        kt = ALPHA * BETA * np.exp(e) * kt**ALPHA
        exact.append(kt)
    assert np.allclose(path["k"], exact, rtol=1e-4)  # second order: errors of third order in the shock
    first = s.first
    linear = [kb + first.Q[k, 0] * 0.05]
    assert abs(path["k"].iloc[0] - exact[0]) < abs(linear[0] - exact[0]) / 10  # far closer than the first order


def test_risk_shows_in_means_and_responses():
    rbc = templates.rbc()
    s = rbc.solve(order=2)
    mean, steady = s.mean(), rbc.steady_state()
    assert mean["k"] > steady["k"]  # precautionary saving: more capital on average than in the steady state
    up = s.irf(12, shock_sd={"e": 0.05}).values[:, :, 0]
    down = s.irf(12, shock_sd={"e": -0.05}).values[:, :, 0]
    assert not np.allclose(up, -down)  # responses depend on the sign of the shock
    small = s.irf(12, shock_sd={"e": 1e-4}).values[:, :, 0] / 1e-4
    linear = rbc.solve().irf(12, shock_sd={"e": 1.0}).values[:, :, 0]
    assert np.allclose(small, linear, rtol=1e-3, atol=1e-6)  # small shocks: the first-order responses
    sim = s.simulate(2000, seed=3, burn=200)
    assert np.isfinite(sim.to_numpy()).all() and abs(sim["k"].mean() - mean["k"]) < 0.02 * mean["k"]
    assert "constant" in s.rules().columns


def test_against_dynare_order_2():
    path = FIXTURES / "dynare-rbc-order2.json"
    if not path.exists():
        pytest.skip("run the dynare-reference workflow with case rbc_order2 and save its output as " + str(path))
    ref = json.loads(path.read_text(encoding="utf-8"))
    s = read_mod(FIXTURES / "dynare" / "rbc_order2.mod").solve(order=2)
    names = ref["endo_names"]
    rows = [names[i - 1] for i in ref["order_var"]]  # Dynare's rows are in its decision-rule order
    states = [names[i - 1] for i in ref["state_var"]]
    assert set(states) == set(s.states)
    ns, k = len(states), len(ref["exo_names"])
    ours_state = [s.states.index(v) for v in states]

    def ours(matrix: np.ndarray, kind: str) -> np.ndarray:
        """Our matrix in Dynare's order of rows and of state (and shock) columns."""
        out = matrix[[s.variables.index(v) for v in rows]]
        if kind == "x":
            return out[:, ours_state]
        if kind == "xx":
            return out.reshape(len(rows), len(s.states), len(s.states))[:, ours_state][:, :, ours_state].reshape(len(rows), ns * ns)
        if kind == "xu":
            return out.reshape(len(rows), len(s.states), k)[:, ours_state].reshape(len(rows), ns * k)
        return out

    def theirs(name: str) -> np.ndarray:
        value = np.array(ref[name], dtype=float)
        return value.reshape(len(rows), -1)

    assert np.allclose(ours(s.ghx, "x"), theirs("ghx"), rtol=1e-10, atol=1e-14)
    assert np.allclose(ours(s.ghu, "u"), theirs("ghu"), rtol=1e-10, atol=1e-14)
    assert np.allclose(ours(s.ghxx, "xx"), theirs("ghxx"), rtol=1e-10, atol=1e-14)
    assert np.allclose(ours(s.ghxu, "xu"), theirs("ghxu"), rtol=1e-10, atol=1e-14)
    assert np.allclose(ours(s.ghuu, "u"), theirs("ghuu"), rtol=1e-10, atol=1e-14)
    assert np.allclose(ours(s.ghs2[:, None], "u"), theirs("ghs2"), rtol=1e-9, atol=1e-15)


def test_the_dialog_and_errors():
    from pymetrix.commands import LoadTemplate, Session, SolveSecondOrder
    from pymetrix.views import view_sections

    s = Session()
    s.run(LoadTemplate("rbc", name="rbc"))
    s.run(SolveSecondOrder("rbc"))
    assert "second = rbc.solve(order=2)" in s.script()
    assert [t for t, _ in view_sections(s["second"])][:2] == ["Decision rules", "Means"]
    with pytest.raises(ValueError, match="order is 1, 2 or 3"):
        templates.rbc().solve(order=4)
