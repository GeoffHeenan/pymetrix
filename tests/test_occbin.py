"""Occasionally binding constraints (OccBin): checked against the nonlinear perfect-foresight solution of the same
model (stacked Newton, a different algorithm) for one shock, anticipated shocks and repeated surprises; against a
direct nonlinear simulation for a backward-looking model with min; and against the linear solution when the
constraint never binds."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, Model, Scenario, occbin, templates
from pymetrix.metadata import Role
from pymetrix.models import ModelError, Newton, PerfectForesight

SHOCKS = ["ey", "epi", "ez"]


@pytest.fixture(scope="module")
def zlb():
    """A New Keynesian model with a persistent natural rate and a zero lower bound on the policy rate."""
    eqs = [Equation("y ~ y[+1] - sig*(i - pi[+1] - rn) + ey", params=["sig"]),
           Equation("pi ~ bet*pi[+1] + kap*y + epi", params=["bet", "kap"]),
           Equation("ishadow ~ rn + phi*pi + 0.5*y", params=["phi"]),
           Equation("i = max(ishadow, 0)", name="lower bound"),
           Equation("rn ~ (1-rho)*rbar + rho*rn[-1] + ez", params=["rho", "rbar"])]
    return Model(eqs, params=dict(sig=1.0, bet=0.99, kap=0.1, phi=1.5, rbar=1.0, rho=0.8),
                 roles={s: Role.SHOCK for s in SHOCKS})


def _perfect_foresight(model, shocks, scenario=None, T=40):
    ss = model.steady_state()
    idx = pd.period_range("2000Q1", periods=T + 61, freq="Q", name="time")
    frame = pd.DataFrame({v: ss[v] for v in model.endogenous}, index=idx)
    for s in SHOCKS:
        frame[s] = 0.0
    for name, path in shocks.items():
        for t, value in path.items():
            frame.loc[idx[t], name] = value
    sol = model.solve(Dataset.from_frame(frame), (idx[1], idx[-1]), scenario, solver=PerfectForesight(terminal="steady_state"))
    return sol.frame.iloc[1:T + 1].reset_index(drop=True), idx


def test_one_shock_equals_perfect_foresight(zlb):
    path = occbin(zlb, {"ez": {1: -2.0}}, periods=40, shock_names=SHOCKS)
    pf, _ = _perfect_foresight(zlb, {"ez": {1: -2.0}})
    for v in ("y", "pi", "i", "rn"):
        assert np.allclose(path.frame[v].to_numpy(), pf[v].to_numpy(), atol=1e-10), v
    assert list(path.binding.index[path.binding["lower bound (max)"]]) == [1, 2, 3, 4]
    assert (path.frame["i"].iloc[:4] == 0).all() and (path.linear["i"].iloc[:4] < 0).all()  # the linear path breaches it
    assert "1-4" in str(path.summary())


def test_anticipated_and_surprise_shocks_equal_perfect_foresight(zlb):
    shocks = {"ez": {1: -1.5, 4: -1.0}}
    known = occbin(zlb, shocks, periods=40, shock_names=SHOCKS, anticipated=True)
    pf, idx = _perfect_foresight(zlb, shocks)  # shocks in the data: known from the start
    assert np.allclose(known.frame["i"], pf["i"], atol=1e-10) and np.allclose(known.frame["y"], pf["y"], atol=1e-10)
    surprises = occbin(zlb, shocks, periods=40, shock_names=SHOCKS)
    scenario = Scenario(set={"ez": {str(idx[1]): -1.5, str(idx[4]): -1.0}}, unanticipated=["ez"])
    pf_s, _ = _perfect_foresight(zlb, {}, scenario)
    for v in ("y", "pi", "i"):
        assert np.allclose(surprises.frame[v], pf_s[v], atol=1e-10), v
    assert not np.allclose(surprises.frame["y"], known.frame["y"])  # news matters


def test_a_min_constraint_in_a_backward_model_equals_simulation():
    model = Model([Equation("x = min(0.9*x[-1] + e, 1)"), Equation("z = 0.5*x")], roles={"e": Role.SHOCK})
    path = occbin(model, {"e": {1: 3.0, 3: 0.5}}, periods=10)
    idx = pd.period_range("2000Q1", periods=11, freq="Q", name="time")
    e = np.zeros(11)
    e[1], e[3] = 3.0, 0.5
    data = Dataset.from_frame(pd.DataFrame({"x": 0.0, "z": 0.0, "e": e}, index=idx))
    exact = model.solve(data, (idx[1], idx[-1]), solver=Newton()).frame.iloc[1:]
    assert np.allclose(path.frame["x"], exact["x"]) and np.allclose(path.frame["z"], exact["z"])
    assert path.binding.iloc[:, 0].sum() >= 1 and path.frame["x"].max() == pytest.approx(1.0)


def test_a_constraint_that_never_binds_gives_the_linear_solution():
    nk = templates.new_keynesian(lower_bound=0.0)
    path = occbin(nk, {"ey": {1: -1.0}}, periods=12)
    irf = nk.solve().irf(11, shock_sd={"ey": 1.0})
    assert not path.binding.any().any()
    assert np.allclose(path.deviations["y"].to_numpy(), -irf["y", "ey"].to_numpy())
    deep = occbin(nk, {"ey": [-4.0, -4.0, -4.0]}, periods=12, anticipated=True)
    assert deep.binding["policy (max)"].sum() >= 1
    assert deep.deviations["i"][deep.binding["policy (max)"]].to_numpy() == pytest.approx(-0.75)  # rate at 0: -(1 + 2)/4


def test_the_unconstrained_path_knows_the_same_shocks():
    """The path without constraints is the counterfactual of the constrained one: with anticipated shocks it is the
    linear model's perfect-foresight path, not a string of surprises (it once was, and so never reached a floor the
    constrained path was held at)."""
    shocks = {"ey": {1: -2.5, 2: -2.5, 3: -2.5, 4: -2.5}}
    path = occbin(templates.new_keynesian(lower_bound=0.0), shocks, periods=20, anticipated=True)
    plain = templates.new_keynesian()  # the same model with no floor: linear, so perfect foresight is the linear path
    ss = plain.model.steady_state()
    idx = pd.period_range("2000Q1", periods=101, freq="Q", name="time")
    frame = pd.DataFrame({v: ss[v] for v in plain.model.endogenous}, index=idx)
    for s in plain.shocks:
        frame[s] = 0.0
    for t, value in shocks["ey"].items():
        frame.loc[idx[t], "ey"] = value
    pf = plain.model.solve(Dataset.from_frame(frame), (idx[1], idx[-1]),
                           solver=PerfectForesight(terminal="steady_state")).frame.iloc[1:21]
    for v in ("y", "pi", "i"):
        assert np.allclose(path.linear[v].to_numpy(), pf[v].to_numpy(), atol=1e-10), v
    binds = path.binding["policy (max)"]
    assert list(binds.index[binds]) == [1, 2, 3]
    assert np.allclose(path.deviations["i"][binds], -0.75)  # held at the floor: a rate of 0 is -(1 + 2)/4
    assert (path.linear["i"][binds] - path.steady["i"] < -0.75).all()  # the rule without the floor goes below it
    surprises = occbin(templates.new_keynesian(lower_bound=0.0), shocks, periods=20)
    plain_irf = plain.solve().irf(19, shock_sd={"ey": 1.0})["y", "ey"].to_numpy()
    expected = sum(-2.5 * np.concatenate([np.zeros(k), plain_irf[:20 - k]]) for k in range(4))  # surprises: the IRFs added
    assert np.allclose(surprises.linear["y"].to_numpy(), expected)


def test_errors(zlb):
    with pytest.raises(ModelError, match="no max or min"):
        occbin(templates.new_keynesian(), {"ey": {1: -1.0}})
    with pytest.raises(ModelError, match="both branches equal"):
        occbin(Model([Equation("x = max(0.5*x[-1] + e, 0)")], roles={"e": Role.SHOCK}), {"e": {1: 1.0}})
    with pytest.raises(ModelError, match="not a shock"):
        occbin(zlb, {"eq": {1: 1.0}}, shock_names=SHOCKS)
    with pytest.raises(ModelError, match="periods run from 1"):
        occbin(zlb, {"ez": {50: 1.0}}, periods=40, shock_names=SHOCKS)


def test_the_dialog():
    from pymetrix.commands import LoadTemplate, OccasionallyBinding, Session
    from pymetrix.commands.occbin import _shocks
    from pymetrix.views import view_sections

    assert _shocks("ey[1]=-3, ez[2:3]=-1") == {"ey": {1: -3.0}, "ez": {2: -1.0, 3: -1.0}}
    s = Session()
    s.run(LoadTemplate("new_keynesian", params="lower_bound=0", name="nk"))
    s.run(OccasionallyBinding("nk", "ey[1:3]=-4", periods=12, anticipated=True))
    assert "constrained = occbin(nk, {'ey': {1: -4.0, 2: -4.0, 3: -4.0}}, periods=12, anticipated=True)" in s.script()
    assert [t for t, _ in view_sections(s["constrained"])][0] == "Summary" and s["constrained"].binding.any().any()
    assert OccasionallyBinding("nk", "ey=1").validate(s)[0].startswith("give shocks as")
    assert OccasionallyBinding("nk", "ey[50]=1", periods=12).validate(s) == ["ey[50] after the last period (12)"]
