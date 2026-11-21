"""Simulation plans on first-order solutions: anticipated and unanticipated shocks, exogenize and endogenize."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, FirstOrder, Model, PerfectForesight, Scenario, templates
from pymetrix.metadata import Role
from pymetrix.models import ModelError

SPAN = ("2025Q1", "2034Q4")


@pytest.fixture(scope="module")
def nk():
    return templates.new_keynesian()


@pytest.fixture(scope="module")
def flat(nk):
    """Every variable at its steady state (zero) from 2024 to 2074, for the perfect-foresight comparisons."""
    idx = pd.period_range("2024Q1", "2074Q4", freq="Q", name="time")
    return Dataset.from_frame(pd.DataFrame({v: 0.0 for v in [*nk.model.endogenous, *nk.model.exogenous]}, index=idx))


def _pf(nk, flat, shocks):
    return nk.model.solve(flat, ("2025Q1", "2074Q4"), Scenario(set=shocks), solver=PerfectForesight(terminal="steady_state"))


def test_an_unanticipated_shock_is_an_impulse_response(nk):
    sol = nk.solve()
    path = sol.solve(None, SPAN, Scenario(set={"ei": {"2025Q1": 1.0}}, unanticipated=["ei"]))
    irf = sol.irf(8)
    for v in ("y", "pi", "i"):
        np.testing.assert_allclose(path[v].loc["2025Q1":"2027Q1"].to_numpy(), irf.values[:, sol.model.observed.index(v), sol.shocks.index("ei")], atol=1e-12)
    assert path.span == SPAN and path.provenance.spec["unanticipated"] == ["ei"]


def test_news_matches_perfect_foresight(nk, flat):
    sol = nk.solve()
    news = sol.solve(None, SPAN, Scenario(set={"ei": {"2026Q1": 1.0}}))  # announced in 2025Q1, happens in 2026Q1
    pf = _pf(nk, flat, {"ei": pd.Series([1.0], index=pd.PeriodIndex(["2026Q1"], freq="Q"))})
    np.testing.assert_allclose(news.frame.loc["2025Q1":"2034Q4", ["y", "pi", "i"]], pf.frame.loc["2025Q1":"2034Q4", ["y", "pi", "i"]], atol=1e-10)
    surprise = sol.solve(None, SPAN, Scenario(set={"ei": {"2026Q1": 1.0}}, unanticipated=["ei"]))
    assert abs(news["i"].loc["2025Q1"]) > 0.1 and surprise["i"].loc["2025Q1":"2025Q4"].abs().max() == 0  # news moves things early
    via_model = nk.model.solve(flat, SPAN, Scenario(set={"ei": {"2026Q1": 1.0}}), solver=FirstOrder())
    np.testing.assert_allclose(via_model["i"].loc["2025Q1":"2034Q4"], news["i"].loc["2025Q1":"2034Q4"], atol=1e-12)


def test_holding_a_variable_with_a_free_shock(nk, flat):
    sol = nk.solve()
    hold = {"2025Q1": 0.5, "2025Q2": 0.5, "2025Q3": 0.5, "2025Q4": 0.5}
    held = sol.solve(None, SPAN, Scenario(hold={"i": hold}, free=["ei"]))  # an announced path for the policy rate
    np.testing.assert_allclose(held["i"].loc["2025Q1":"2025Q4"], 0.5, atol=1e-12)
    assert abs(held["i"].loc["2026Q1"] - 0.5) > 0.05  # free again afterwards
    assert (held["ei"].loc["2026Q1":] == 0).all()  # the shock is only used where the target is set
    replay = _pf(nk, flat, {"ei": held["ei"].loc["2025Q1":"2025Q4"]})  # the solved shocks, given to an independent solver
    np.testing.assert_allclose(replay["i"].loc["2025Q1":"2025Q4"], 0.5, atol=1e-9)
    # unanticipated: each period's shock, chosen when it happens, has a closed form
    surprise = sol.solve(None, SPAN, Scenario(hold={"i": {"2025Q1": 0.5, "2025Q2": 0.5}}, free=["ei"], unanticipated=["ei"]))
    j, k, x, expected = sol.variables.index("i"), sol.shocks.index("ei"), np.zeros(len(sol.variables)), []
    for _ in range(2):
        e = (0.5 - (sol.P @ x)[j]) / sol.Q[j, k]
        x = sol.P @ x + sol.Q[:, k] * e
        expected.append(e)
    np.testing.assert_allclose(surprise["ei"].loc["2025Q1":"2025Q2"], expected, rtol=1e-10)
    assert not np.allclose(surprise["ei"].loc["2025Q1":"2025Q2"], held["ei"].loc["2025Q1":"2025Q2"])


def test_starting_values_lags_and_a_steady_state():
    model = Model([Equation("x ~ c + a*x[-1] + b*x[-2] + e", params=["c", "a", "b"])], params={"c": 1.0, "a": 0.5, "b": 0.2},
                  roles={"e": Role.SHOCK})
    idx = pd.period_range("2020Q1", "2024Q4", freq="Q", name="time")
    data = Dataset.from_frame(pd.DataFrame({"x": np.linspace(2.0, 4.0, len(idx)), "e": 0.0}, index=idx))
    path = model.solve(data, ("2025Q1", "2026Q4"), solver=FirstOrder())
    x = list(data["x"])
    for _ in range(8):
        x.append(1.0 + 0.5 * x[-1] + 0.2 * x[-2])  # the model, by hand: exact because it is linear
    np.testing.assert_allclose(path["x"].loc["2025Q1":"2026Q4"], x[-8:], rtol=1e-12)
    assert path["x"].loc["2024Q4"] == 4.0  # history kept
    from_steady = model.linearize(shocks=["e"]).solve().solve(None, ("2025Q1", "2025Q4"))
    np.testing.assert_allclose(from_steady["x"].loc["2025Q1":"2025Q4"], 1.0 / 0.3)


def test_a_nonlinear_model_near_its_steady_state():
    rbc = templates.rbc()
    ss = rbc.model.steady_state(guess=getattr(rbc, "guess", None))
    idx = pd.period_range("2024Q1", "2064Q4", freq="Q", name="time")
    flat = Dataset.from_frame(pd.DataFrame({k: v for k, v in ss.values.items()}, index=idx))
    shock = {rbc.model.exogenous[0]: {"2025Q1": 0.001}}
    first = rbc.model.solve(flat, ("2025Q1", "2064Q4"), Scenario(set=shock, unanticipated=[rbc.model.exogenous[0]]), solver=FirstOrder())
    exact = rbc.model.solve(flat, ("2025Q1", "2064Q4"), Scenario(set=shock), solver=PerfectForesight(terminal="steady_state"))
    for v in rbc.model.endogenous:  # a surprise in the first period is the same as an announcement then
        dev_first = first[v].loc["2025Q1":"2030Q4"] - ss[v]
        dev_exact = exact[v].loc["2025Q1":"2030Q4"] - ss[v]
        assert np.max(np.abs(dev_first - dev_exact)) < 0.01 * np.max(np.abs(dev_exact)) + 1e-12, v


def test_errors(nk):
    sol = nk.solve()
    with pytest.raises(ModelError, match="pair each hold= with one free= shock"):
        sol.solve(None, SPAN, Scenario(hold={"i": 0.5}))
    with pytest.raises(ModelError, match="free=: y are not shocks"):
        sol.solve(None, SPAN, Scenario(hold={"i": 0.5}, free=["y"]))
    with pytest.raises(ModelError, match="need a nonlinear solver"):
        sol.solve(None, SPAN, Scenario(fix={"i": None}))
    with pytest.raises(ModelError, match="outside the span"):
        sol.solve(None, SPAN, Scenario(set={"ei": {"2040Q1": 1.0}}))
    idx = pd.period_range("2020Q1", "2024Q4", freq="Q", name="time")
    with pytest.raises(ModelError, match="starting values are missing for i in 2024Q4"):  # i[-1] is in the model
        sol.solve(Dataset.from_frame(pd.DataFrame({"pi": 0.0, "y": 0.0}, index=idx)), SPAN)
    sol.solve(Dataset.from_frame(pd.DataFrame({"i": 0.0}, index=idx)), SPAN)  # y and pi enter unlagged: not needed
    two = Model([Equation("a ~ r*a[-1] + ea", params=["r"]), Equation("b ~ r*b[-1] + eb", params=["r"])], params={"r": 0.5},
                roles={"ea": Role.SHOCK, "eb": Role.SHOCK})
    with pytest.raises(ModelError, match="cannot put the targets on their paths"):
        two.linearize().solve().solve(None, ("2025Q1", "2025Q4"), Scenario(hold={"a": {"2025Q2": 1.0}}, free=["eb"]))
    backward = Dataset.from_frame(pd.DataFrame({"a": 0.0, "b": 0.0, "ea": 0.0, "eb": 0.0},
                                               index=pd.period_range("2024Q1", "2025Q4", freq="Q", name="time")))
    surprise = two.solve(backward, ("2025Q1", "2025Q4"), Scenario(set={"ea": {"2025Q2": 1.0}}, unanticipated=["ea"]))
    plain = two.solve(backward, ("2025Q1", "2025Q4"), Scenario(set={"ea": {"2025Q2": 1.0}}))
    np.testing.assert_allclose(surprise["a"], plain["a"])  # no expectations in the model: anticipation changes nothing


def test_the_dialog_records_the_plan(nk, tmp_path):
    from pymetrix.commands import DefineEquation, Session, SolveModel
    from pymetrix.testing import check_project

    s = Session()
    idx = pd.period_range("2020Q1", "2024Q4", freq="Q", name="time")
    s.objects["data"] = Dataset.from_frame(pd.DataFrame({"y": 0.0, "pi": 0.0, "i": 0.0, "ey": 0.0, "epi": 0.0, "ei": 0.0}, index=idx))
    values = nk.model.parameters.values()
    for k, eq in enumerate(e.text for e in nk.model.equations):
        s.run(DefineEquation(eq, params=", ".join(p for p in values if p in eq), name=f"eq{k}"))
    check_project(s, SolveModel("eq0, eq1, eq2", "2025Q1", "2026Q4", solver="first_order", params=", ".join(f"{p}={v}" for p, v in values.items()),
                                hold="i=0.5", free="ei", unanticipated="ei", name="held"), tmp_path)
    assert "scenario=Scenario(hold={'i': 0.5}, free=['ei'], unanticipated=['ei'])" in s.script()
    np.testing.assert_allclose(s["held"]["i"].loc["2025Q1":"2026Q4"], 0.5, atol=1e-12)



def test_the_dialog_takes_values_for_some_periods(nk, tmp_path):
    from pymetrix.commands import DefineEquation, Session, SolveModel
    from pymetrix.testing import check_project

    s = Session()
    idx = pd.period_range("2020Q1", "2024Q4", freq="Q", name="time")
    s.objects["data"] = Dataset.from_frame(pd.DataFrame({"y": 0.0, "pi": 0.0, "i": 0.0, "ey": 0.0, "epi": 0.0, "ei": 0.0}, index=idx))
    values = nk.model.parameters.values()
    for k, eq in enumerate(e.text for e in nk.model.equations):
        s.run(DefineEquation(eq, params=", ".join(p for p in values if p in eq), name=f"eq{k}"))
    check_project(s, SolveModel("eq0, eq1, eq2", "2025Q1", "2028Q4", solver="first_order", params=", ".join(f"{p}={v}" for p, v in values.items()),
                                set_to="ey[2025Q1]=-1, ey[2025Q2]=-0.5", hold="i[2025Q1:2025Q4]=-0.5", free="ei", name="guidance"), tmp_path)
    path = s["guidance"]
    np.testing.assert_allclose(path["i"].loc["2025Q1":"2025Q4"], -0.5, atol=1e-12)
    assert abs(path["i"].loc["2026Q1"] + 0.5) > 1e-3 and path["ey"].loc["2025Q2"] == -0.5 and path["ey"].loc["2025Q3"] == 0
    assert "hold={'i': {'2025Q1': -0.5, '2025Q2': -0.5, '2025Q3': -0.5, '2025Q4': -0.5}}" in s.script()
    assert SolveModel("eq0", "2025Q1", "2025Q4", hold="i(2025)=1").validate(s)[0].startswith("scenario: write name=value")
    from pymetrix.commands.registry import from_json  # a step saved at schema 1, when hold was called target

    old = from_json({"kind": "command", "id": "pymetrix:solve_model", "version": 1,
                     "fields": {"equations": "eq0", "start": "2025Q1", "end": "2025Q4", "target": "i[2025Q1:2025Q4]=-0.5"}})
    assert old == SolveModel("eq0", "2025Q1", "2025Q4", hold="i[2025Q1:2025Q4]=-0.5")


def test_theoretical_moments_and_variance_decomposition(nk):
    from pymetrix.models import ModelError

    ar = Model([Equation("x ~ rho*x[-1] + e", params=["rho"])], params={"rho": 0.7}, roles={"e": Role.SHOCK})
    m = ar.linearize().solve().moments({"e": 0.5}, lags=3)  # the AR(1) closed form: var = s^2 / (1 - rho^2), acf rho^k
    assert m.std["x"] == pytest.approx(0.5 / np.sqrt(1 - 0.49)) and m.autocorrelation.loc["x"].tolist() == pytest.approx([0.7, 0.49, 0.343])
    sol = nk.solve()
    moments = sol.moments(nk.shock_sd)
    impact = sol.Q * np.array([nk.shock_sd[s] for s in sol.shocks])  # independently: the sum over impulse responses
    S, x = np.zeros_like(sol.P), impact
    for _ in range(2000):
        S += x @ x.T
        x = sol.P @ x
    np.testing.assert_allclose(moments.covariance.to_numpy(), S[:3, :3], rtol=1e-10)
    assert np.allclose(moments.shares.sum(axis=1), 1.0)
    fevd = sol.fevd(200, nk.shock_sd)  # long horizons converge to the unconditional shares
    np.testing.assert_allclose(fevd.values[-1], moments.shares.to_numpy(), atol=1e-10)
    np.testing.assert_allclose(fevd.values[0].sum(axis=1), 1.0)
    simulated = sol.simulate(200_000, nk.shock_sd, seed=3, levels=False)  # a sanity check, by simulation
    assert np.allclose(simulated.std().to_numpy(), moments.std.to_numpy(), rtol=0.02)
    walk = Model([Equation("x = x[-1] + e")], roles={"e": Role.SHOCK})
    with pytest.raises(ModelError, match="unit root"):
        walk.linearize(walk.steady_state(growth=True), shocks=["e"]).solve().moments()
    assert walk.linearize(walk.steady_state(growth=True), shocks=["e"]).solve().fevd(4).values.shape == (4, 1, 1)


def test_shock_decompositions_recover_the_shocks():
    model = Model([Equation("x ~ a*x[-1] + b*x[-2] + e + 0.5*u", params=["a", "b"]), Equation("z ~ c*z[-1] + x + u", params=["c"])],
                  params={"a": 0.5, "b": 0.2, "c": 0.3}, roles={"e": Role.SHOCK, "u": Role.SHOCK})
    sol = model.linearize().solve()
    rng = np.random.default_rng(0)
    shocks = rng.normal(size=(60, 2))
    x = np.zeros(len(sol.variables))
    rows = []
    for t in range(60):  # a path made from known shocks
        x = sol.P @ x + sol.Q @ shocks[t]
        rows.append(x[:2].copy())
    levels = pd.DataFrame(rows, columns=list(sol.model.observed), index=pd.period_range("2000Q1", periods=60, freq="Q", name="time"))
    d = sol.decompose(levels)
    first = len(levels) - len(d.deviations)  # x[-2] needs one period before the first state
    np.testing.assert_allclose(d.shocks.to_numpy()[1:], shocks[first + 1:], atol=1e-10)  # the same shocks, recovered
    assert float(d.residual.abs().max().max()) < 1e-12  # contributions + initial conditions = the path
    assert list(d.contributions["z"].columns) == ["e", "u", "initial conditions"]
    assert "Shock decomposition: z" in str(d.table("z"))
    from pymetrix.views import titles

    assert {"Table: x", "Table: z", "Shocks"} <= set(titles(d))  # and a Graph tab when matplotlib is installed



def test_surprises_in_nonlinear_perfect_foresight(nk, flat):
    sol = nk.solve()
    shocks = {"ey": {"2026Q1": 1.0}, "ei": {"2025Q3": 0.5}}
    plan = Scenario(set=shocks, unanticipated=["ei"])  # the demand shock announced at the start; the rate shock a surprise
    linear = sol.solve(None, ("2025Q1", "2074Q4"), plan)
    stacked = nk.model.solve(flat, ("2025Q1", "2074Q4"), plan, solver=PerfectForesight(terminal="steady_state"))
    for v in ("y", "pi", "i"):  # a linear model: re-solving at the surprise gives the first-order plan exactly
        np.testing.assert_allclose(stacked[v].loc["2025Q1":"2034Q4"], linear[v].loc["2025Q1":"2034Q4"], atol=1e-9)
    assert stacked.provenance.method["expectation errors at"] == ["2025Q3"]
    unknown = nk.model.solve(flat, ("2025Q1", "2074Q4"), Scenario(set={"ey": {"2026Q1": 1.0}}), solver=PerfectForesight(terminal="steady_state"))
    np.testing.assert_allclose(stacked.frame.loc["2025Q1":"2025Q2", ["y", "pi", "i"]],  # before the surprise, nobody knew
                               unknown.frame.loc["2025Q1":"2025Q2", ["y", "pi", "i"]], atol=1e-10)
    assert not np.allclose(stacked["i"].loc["2025Q3"], unknown["i"].loc["2025Q3"])


def test_a_surprise_at_the_lower_bound():
    eqs = [Equation("y ~ y[+1] - (1/sigma)*(i - pi[+1]) + ey", params=["sigma"]),
           Equation("pi ~ beta*pi[+1] + kappa*y + epi", params=["beta", "kappa"]),
           Equation("iu ~ rho*iu[-1] + (1-rho)*(phi_pi*pi + phi_y*y) + ei", params=["rho", "phi_pi", "phi_y"]),
           Equation("i ~ max(iu, floor)", params=["floor"])]
    model = Model(eqs, params=dict(sigma=1, beta=0.99, kappa=0.1, rho=0.8, phi_pi=1.5, phi_y=0.5, floor=-0.25),
                  roles={"ey": Role.SHOCK, "epi": Role.SHOCK, "ei": Role.SHOCK})
    idx = pd.period_range("2024Q1", "2054Q4", freq="Q", name="time")
    flat = Dataset.from_frame(pd.DataFrame({v: 0.0 for v in [*model.endogenous, *model.exogenous]}, index=idx))
    span = ("2025Q1", "2054Q4")
    surprise = model.solve(flat, span, Scenario(set={"ey": {"2025Q2": -3.0}}, unanticipated=["ey"]), solver=PerfectForesight(terminal="steady_state"))
    news = model.solve(flat, span, Scenario(set={"ey": {"2025Q2": -3.0}}), solver=PerfectForesight(terminal="steady_state"))
    assert surprise["i"].loc["2025Q1"] == pytest.approx(0.0, abs=1e-10)  # a surprise: nothing happens before it
    assert news["i"].loc["2025Q1"] < -0.1  # announced: the economy moves at once
    np.testing.assert_allclose(surprise["i"].loc["2025Q1":"2034Q4"], np.maximum(surprise["iu"].loc["2025Q1":"2034Q4"], -0.25), atol=1e-10)
    assert surprise["iu"].loc["2025Q2"] < -0.25 and surprise["i"].loc["2025Q2"] == pytest.approx(-0.25)  # the surprise hits the bound


def test_scenario_target_is_now_hold():
    from pymetrix.codec import decode, encode

    with pytest.warns(DeprecationWarning, match="'target' of Scenario\\(\\) is now called 'hold'"):
        old = Scenario(target={"i": 0.5}, free=["ei"])
    assert old.hold == {"i": 0.5} and not hasattr(old, "target")
    with pytest.raises(TypeError, match="both"):
        Scenario(target={"i": 0.5}, hold={"i": 0.5})
    saved = encode(Scenario(hold={"i": 0.5}, free=["ei"]))  # a project saved before the rename has target
    saved["target"] = saved.pop("hold")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # reading an old project warns about nothing
        assert decode(saved).hold == {"i": 0.5}
