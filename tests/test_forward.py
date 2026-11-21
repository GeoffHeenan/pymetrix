"""Forward-looking models: perfect foresight, steady state, linearisation and the rational-expectations solution."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, Model, Newton, PerfectForesight
from pymetrix.models import LinearSolution, ModelError


def _quarters(n, **columns):
    idx = pd.period_range("2000Q1", periods=n, freq="Q")
    return Dataset.from_frame(pd.DataFrame({"t": idx, **columns}), time="t"), idx


def _nk(phi=1.5):
    """A three-equation New Keynesian model: IS curve, Phillips curve, Taylor rule with smoothing."""
    return Model([Equation("y ~ y[+1] - s*(i - pi[+1]) + ey", params=["s"]),
                  Equation("pi ~ b*pi[+1] + k*y + epi", params=["b", "k"]),
                  Equation("i ~ r*i[-1] + (1-r)*(phi*pi + 0.5*y) + ei", params=["r", "phi"])],
                 params=dict(s=1.0, b=0.99, k=0.1, r=0.8, phi=phi))


def test_perfect_foresight_matches_the_analytic_forward_solution():
    z = np.zeros(20)
    z[10] = 1.0
    data, idx = _quarters(20, x=0.0, z=z)
    model = Model([Equation("x = 0.5*x[+1] + z")])
    assert model.has_leads
    sol = model.solve(data, (idx[1], idx[-1]), solver=PerfectForesight(terminal="steady_state"))
    x = sol.frame["x"].to_numpy()
    expected = np.r_[0.0, [0.5 ** (10 - t) for t in range(1, 11)], np.zeros(9)]  # anticipated: x(t) = 0.5^(10-t)
    assert np.allclose(x, expected, atol=1e-10)


def test_perfect_foresight_equals_newton_on_a_backward_model():
    rng = np.random.default_rng(0)
    data, idx = _quarters(30, y=1.0, g=rng.normal(size=30) + 2)
    model = Model([Equation("y = 0.6*y[-1] + 0.3*g + 0.05*y*g")])
    span = (idx[1], idx[-1])
    stacked = model.solve(data, span, solver=PerfectForesight()).frame["y"]
    period = model.solve(data, span, solver=Newton()).frame["y"]
    assert np.allclose(stacked, period, atol=1e-9)


def test_perfect_foresight_terminal_conditions():
    data, idx = _quarters(12, x=0.0, z=1.0)
    model = Model([Equation("x = 0.5*x[+1] + z")])
    constant = model.solve(data, (idx[0], idx[-1]), solver=PerfectForesight()).frame["x"]
    assert np.allclose(constant, 2.0)  # x = 0.5 x + 1 at the end, and so throughout
    gap = data.edit({"x": {idx[-1]: None}})
    with pytest.raises(ModelError, match="not in the data"):
        model.solve(gap, (idx[0], idx[-2]), solver=PerfectForesight(terminal="data"))
    with pytest.raises(ModelError, match="all periods together"):
        PerfectForesight().solve_period(None)


def test_steady_state():
    model = Model([Equation("c = 0.8*y + a"), Equation("y = c + g")])
    ss = model.steady_state({"g": 2.0, "a": 1.0})
    assert ss["y"] == pytest.approx(15.0) and ss["c"] == pytest.approx(13.0) and ss["g"] == 2.0
    assert "Steady state" in str(ss)
    with pytest.raises(ModelError, match="no steady state"):
        Model([Equation("x = x[-1] + 1")]).steady_state()


def test_rational_expectations_solution_of_the_new_keynesian_model():
    sol = _nk().linearize(shocks=["ey", "epi", "ei"]).solve()
    assert isinstance(sol, LinearSolution) and sol.variables == ("y", "pi", "i")
    lin, P, Q = sol.model, sol.P, sol.Q
    assert np.allclose(lin.A0 @ P @ P + lin.A1 @ P + lin.A2, 0, atol=1e-10)  # the solution satisfies the model
    assert np.allclose((lin.A0 @ P + lin.A1) @ Q + lin.B, 0, atol=1e-10)
    assert np.allclose(P[:, :2], 0)  # only the lagged interest rate is a state
    irf = sol.irf(12)
    assert irf.values[0, 0, 2] < 0 and irf.values[0, 1, 2] < 0 and irf.values[0, 2, 2] > 0  # tightening: output and inflation fall
    assert abs(irf.values[-1, 2, 2]) < 0.01
    assert np.sum(np.abs(sol.eigenvalues) < 1) == 1  # one stable root per lagged variable (only i)
    assert list(sol.rules().columns)[1:] == ["i[-1]", "ey", "epi", "ei"]
    sim = sol.simulate(200, {"ei": 0.25}, seed=1)
    assert sim.shape == (200, 3) and 0 < sim["i"].std() < 2


def test_blanchard_kahn_failures_are_explained():
    with pytest.raises(ModelError, match="not unique.*Taylor"):
        _nk(phi=0.5).linearize(shocks=["ei"]).solve()
    with pytest.raises(ModelError, match="no stable solution"):
        Model([Equation("x = 0.5*x[-1] + 0.6*x[-2] + e")]).linearize(shocks=["e"]).solve()
    with pytest.raises(ModelError, match="must be exogenous"):
        _nk().linearize(shocks=["y"])


def test_one_equation_and_longer_leads_and_lags():
    # x = 0.5 E x[+1] + e with e serially uncorrelated: x responds to e one for one, and not after
    sol = Model([Equation("x = 0.5*x[+1] + e")]).linearize(shocks=["e"]).solve()
    assert sol.irf(3).values[:, 0, 0] == pytest.approx([1, 0, 0, 0])
    # x = 0.4 x[-1] + 0.2 x[-2] + 0.2 E x[+2] + e: auxiliary variables, and the path satisfies the model
    sol = Model([Equation("x = 0.4*x[-1] + 0.2*x[-2] + 0.2*x[+2] + e")]).linearize(shocks=["e"]).solve()
    assert sol.variables == ("x", "x_lag1", "x_lead1")
    v = sol.irf(40).values[:, 0, 0]
    assert max(abs(v[t] - (0.4 * v[t - 1] + 0.2 * v[t - 2] + 0.2 * v[t + 2])) for t in range(2, 30)) < 1e-12
    assert list(sol.rules().columns)[1:] == ["x[-1]", "x[-2]", "e"]


def test_perfect_foresight_agrees_with_the_linear_solution():
    model = _nk()
    irf = model.linearize(shocks=["ei"]).solve().irf(9)
    data, idx = _quarters(60, y=0.0, pi=0.0, i=0.0, ey=0.0, epi=0.0, ei=np.r_[0.0, 1.0, np.zeros(58)])
    pf = model.solve(data, (idx[1], idx[-1]), solver=PerfectForesight(terminal="steady_state"))
    assert np.allclose(pf.frame[["y", "pi", "i"]].to_numpy()[1:11], irf.values[:, :, 0], atol=1e-10)


def test_kalman_filter_recovers_the_model_variables():
    sd = {"ey": 0.5, "ei": 0.2}  # two shocks, two observables: the output gap is identified
    sol = _nk().linearize(shocks=list(sd)).solve()
    sim = sol.simulate(120, sd, seed=3)
    data, idx = _quarters(120, pi=sim["pi"].to_numpy(), i=sim["i"].to_numpy())
    est = sol.filter(data, ["pi", "i"], sd)
    assert np.allclose(est["i"], sim["i"], atol=1e-6)  # observed exactly
    assert np.allclose(est["y"], sim["y"], atol=1e-6)  # the unobserved output gap, recovered


def test_calibrated_models_need_their_parameters():
    with pytest.raises(ModelError, match="needs values for b"):
        Model([Equation("pi ~ b*pi[+1] + y", params=["b"])])


def test_dialogs_record_and_replay(tmp_path):
    from pymetrix.commands import Session, load_project, save_project
    from pymetrix.commands.builtin import DefineEquation, SolveModel, SolveRationalExpectations

    s = Session()
    s.run(DefineEquation("y ~ y[+1] - s*(i - pi[+1]) + ey", "s", name="eq_is"))
    s.run(DefineEquation("pi ~ b*pi[+1] + k*y + epi", "b, k", name="eq_pc"))
    s.run(DefineEquation("i ~ r*i[-1] + (1-r)*(phi*pi + 0.5*y) + ei", "r, phi", name="eq_tr"))
    calibration = "s=1, b=0.99, k=0.1, r=0.8, phi=1.5"
    cmd = SolveRationalExpectations("eq_is, eq_pc, eq_tr", "ey, epi, ei", params=calibration)
    assert cmd.validate(s) == []
    s.run(cmd)
    assert ("re = Model([eq_is, eq_pc, eq_tr], params={'s': 1.0, 'b': 0.99, 'k': 0.1, 'r': 0.8, 'phi': 1.5})"
            ".linearize(shocks=['ey', 'epi', 'ei']).solve()") in s.script()
    assert isinstance(s["re"], LinearSolution)
    assert SolveRationalExpectations("eq_is", "").validate(s) == ["give at least one shock"]
    data, idx = _quarters(40, y=0.0, pi=0.0, i=0.0, ey=0.0, epi=0.0, ei=np.r_[0.0, 1.0, np.zeros(38)])
    s.objects["data"] = data
    s.run(SolveModel("eq_is, eq_pc, eq_tr", "2000Q2", "2009Q4", solver="perfect_foresight",
                     solver_settings="terminal='steady_state'", params=calibration))
    assert "solver=PerfectForesight(terminal='steady_state')" in s.script()
    reopened = load_project(save_project(s, tmp_path / "fl.pymx"))
    assert np.allclose(reopened["re"].P, s["re"].P)
    assert np.allclose(reopened["solution"].frame["i"], s["solution"].frame["i"])


def test_views():
    pytest.importorskip("matplotlib")
    from pymetrix.ui.render import view_sections

    sol = _nk().linearize(shocks=["ey", "ei"]).solve()
    assert [t for t, _ in view_sections(sol)] == ["Decision rules", "Graph", "Steady state", "Roots"]
    assert len(sol.plot(8).panels) == 6
