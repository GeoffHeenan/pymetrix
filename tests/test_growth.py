"""Steady states with growth: balanced growth paths, undetermined levels, unit roots, and simulation along the path."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, FirstOrder, Model, Scenario
from pymetrix.expr import evaluate
from pymetrix.metadata import Role
from pymetrix.models import ModelError

PARAMS = dict(rho_g=0.9, g_ss=2.0, a1=0.7, a2=0.2, r_bar=1.0, b1=0.5, b2=0.15, c1=0.7, c2=1.5, c3=0.5, pi_ss=2.0)


def gap_trend() -> Model:
    """A small gap-trend model: log output (x100) is potential plus the gap; potential grows at a trend rate."""
    eqs = [
        Equation("l_y = l_y_bar + y_gap", name="output"),
        Equation("l_y_bar = l_y_bar[-1] + g_bar/4", name="potential"),
        Equation("g_bar ~ rho_g*g_bar[-1] + (1 - rho_g)*g_ss + e_g", params=["rho_g", "g_ss"], name="trend_growth"),
        Equation("y_gap ~ a1*y_gap[-1] - a2*(r - r_bar) + e_y", params=["a1", "a2", "r_bar"], name="is_curve"),
        Equation("pi ~ b1*pi[-1] + (1 - b1)*pi[+1] + b2*y_gap + e_pi", params=["b1", "b2"], name="phillips"),
        Equation("i ~ c1*i[-1] + (1 - c1)*(r_bar + pi[+1] + c2*(pi - pi_ss) + c3*y_gap) + e_i",
                 params=["c1", "c2", "c3", "r_bar", "pi_ss"], name="policy"),
        Equation("r = i - pi[+1]", name="real_rate"),
        Equation("dl_y = 4*(l_y - l_y[-1])", name="growth"),
    ]
    return Model(eqs, params=PARAMS, roles={k: Role.SHOCK for k in ("e_g", "e_y", "e_pi", "e_i")})


def test_the_balanced_growth_path():
    model = gap_trend()
    with pytest.raises(ModelError, match="no steady state"):
        model.steady_state()  # no constant steady state: potential output grows
    ss = model.steady_state(growth=True, guess={"l_y": 460, "l_y_bar": 460})
    expected = {"g_bar": 2.0, "y_gap": 0.0, "pi": 2.0, "i": 3.0, "r": 1.0, "dl_y": 2.0}
    assert {k: ss[k] for k in expected} == pytest.approx(expected)
    assert ss.growth == pytest.approx({"l_y": 0.5, "l_y_bar": 0.5})  # 2 percent a year, a quarter at a time
    assert ss.at("l_y", 4) == pytest.approx(ss["l_y"] + 2.0)
    assert set(ss.undetermined) == {"l_y", "l_y_bar"}  # any level growing at the right rate will do
    pinned = model.steady_state(growth=True, fix={"l_y": 460.0})
    assert pinned["l_y"] == 460.0 and pinned["l_y_bar"] == pytest.approx(460.0) and pinned.undetermined == ()
    assert "not determined" in str(ss.table()) and "growth per period" in str(ss.table())


def test_simulations_follow_the_model_along_the_path():
    model = gap_trend()
    ss = model.steady_state(growth=True, guess={"l_y": 460, "l_y_bar": 460})
    sol = model.linearize(ss).solve()  # unit roots: the levels of output and potential
    assert np.sum(np.isclose(np.abs(sol.eigenvalues), 1.0, atol=1e-6)) >= 1
    span = ("2025Q1", "2034Q4")
    path = sol.solve(None, span, Scenario(set={"e_y": {"2025Q1": -1.0}, "e_g": {"2026Q1": 0.5}}))
    frame = path.frame
    periods = frame.loc["2025Q2":"2034Q3"].index  # inside the span: every lag and lead is simulated
    for eq in model.model_equations:  # the model's own equations, in levels, hold on the simulated path
        for p in periods:
            value = evaluate(eq.residual, lambda name, lag, p=p: float(frame.at[p + lag, name]))
            assert abs(value) < 1e-9, (eq.name, str(p), value)
    quiet = sol.solve(None, span)  # no shocks: exactly the growth path
    assert np.allclose(np.diff(quiet["l_y"].loc[span[0]:span[1]]), 0.5) and np.allclose(quiet["dl_y"].loc[span[0]:span[1]], 2.0)


def test_starting_from_data_and_the_solver_option():
    model = gap_trend()
    idx = pd.period_range("2023Q1", "2024Q4", freq="Q", name="time")
    history = {"l_y": 470 + 0.5 * np.arange(8) - 0.5, "l_y_bar": 470 + 0.5 * np.arange(8), "g_bar": 2.0, "y_gap": -0.5,
               "pi": 2.5, "i": 3.0, "r": 1.0, "dl_y": 2.0, "e_g": 0.0, "e_y": 0.0, "e_pi": 0.0, "e_i": 0.0}
    data = Dataset.from_frame(pd.DataFrame(history, index=idx))
    path = model.solve(data, ("2025Q1", "2026Q4"), solver=FirstOrder(growth=True))
    assert path["l_y_bar"].loc["2025Q1"] == pytest.approx(473.5 + 0.5)  # potential carries on from its last value
    assert path["y_gap"].loc["2025Q1"] > -0.5 and path["y_gap"].loc["2025Q1"] < 0  # the gap closes
    with pytest.raises(ModelError, match="no steady state"):
        model.solve(data, ("2025Q1", "2026Q4"), solver=FirstOrder())  # without growth there is none


def test_unit_roots_and_their_limits():
    walk = Model([Equation("x = x[-1] + e")], roles={"e": Role.SHOCK})
    sol = walk.linearize(walk.steady_state(growth=True), shocks=["e"]).solve()
    assert sol.P[0, 0] == pytest.approx(1.0) and np.allclose(sol.irf(6).values[:, 0, 0], 1.0)  # a shock stays
    explosive = Model([Equation("x ~ a*x[-1] + e", params=["a"])], params={"a": 1.1}, roles={"e": Role.SHOCK})
    with pytest.raises(ModelError, match="no stable solution"):
        explosive.linearize(shocks=["e"]).solve()
    accelerating = Model([Equation("x = x[-1] + z"), Equation("z = z[-1] + 1")])
    with pytest.raises(ModelError, match="no steady state with growth"):
        accelerating.steady_state(growth=True)  # x would grow at a growing rate: no balanced path
    with pytest.raises(ModelError, match="not endogenous"):
        gap_trend().steady_state(growth=True, fix={"g_ss": 2.0})


def test_calibration_matches_the_closed_form():
    from pymetrix import templates

    rbc = templates.rbc().model  # 1 = beta (1 + alpha k^(alpha-1) - delta), y = k^alpha: k/y = alpha / (1/beta - 1 + delta)
    one = rbc.calibrate({"k/y": 10}, ["beta"])
    assert one.values["beta"] == pytest.approx(1 / (0.33 / 10 + 1 - 0.025), rel=1e-10)
    assert one.achieved["k/y"] == pytest.approx(10) and one.model.parameters["beta"].value == pytest.approx(one.values["beta"])
    two = rbc.calibrate({"k/y": 10, "c/y": 0.8}, ["beta", "delta"])  # c/y = 1 - delta k/y in the steady state
    assert two.values["delta"] == pytest.approx(0.02, rel=1e-9)
    assert two.values["beta"] == pytest.approx(1 / (0.33 / 10 + 1 - 0.02), rel=1e-9)
    assert "target: k/y" in str(two.summary())
    with pytest.raises(ModelError, match="one parameter for each target"):
        rbc.calibrate({"k/y": 10}, ["beta", "delta"])
    with pytest.raises(ModelError, match="not parameters"):
        rbc.calibrate({"k/y": 10}, ["gamma"])
    with pytest.raises(ModelError, match="not variables"):
        rbc.calibrate({"k/q": 10}, ["beta"])


def test_calibrating_a_trending_model_and_the_dialog(tmp_path):
    model = gap_trend()
    cal = model.calibrate({"i": 3.5, "dl_y": 2.5}, ["r_bar", "g_ss"], growth=True, guess={"l_y": 460, "l_y_bar": 460})
    assert cal.values == pytest.approx({"r_bar": 1.5, "g_ss": 2.5})  # i = r_bar + pi_ss; growth = g_ss
    assert cal.steady.growth["l_y"] == pytest.approx(2.5 / 4)

    from pymetrix.commands import Calibrate, DefineEquation, Session
    from pymetrix.testing import check_project
    from pymetrix.views import titles

    s = Session()
    for k, eq in enumerate(["c ~ c[+1] / (beta*(1 + alpha*k^(alpha - 1) - delta))", "y ~ exp(e)*k[-1]^alpha", "k ~ y + (1 - delta)*k[-1] - c"]):
        s.run(DefineEquation(eq, params=", ".join(p for p in ("alpha", "beta", "delta") if p in eq), name=f"eq{k}"))
    command = Calibrate("eq0, eq1, eq2", targets="k/y = 10; c/y = 0.8", free="beta, delta", params="alpha=0.33, beta=0.99, delta=0.025")
    check_project(s, command, tmp_path)
    assert "calibration = Model([eq0, eq1, eq2], params={'alpha': 0.33, 'beta': 0.99, 'delta': 0.025}).calibrate({'k/y': 10.0, 'c/y': 0.8}, ['beta', 'delta'])" in s.script()
    assert s["calibration"].values["delta"] == pytest.approx(0.02)
    assert titles(s["calibration"])[:3] == ["Summary", "Steady state", "Parameters"]
    assert Calibrate("eq0", targets="k/y = 10", free="beta, delta").validate(s) == ["1 targets but 2 free parameters: give one parameter for each target"]


def test_filtering_and_estimating_along_the_trend():
    model = gap_trend()
    ss = model.steady_state(growth=True, guess={"l_y": 460, "l_y_bar": 460})
    sol = model.linearize(ss).solve()
    rng = np.random.default_rng(5)
    span = ("1990Q1", "2024Q4")
    periods = pd.period_range(*span, freq="Q", name="time")
    shocks = {"e_y": {str(p): v for p, v in zip(periods, rng.normal(0, 0.5, len(periods)))},
              "e_pi": {str(p): v for p, v in zip(periods, rng.normal(0, 0.3, len(periods)))},
              "e_g": {str(p): v for p, v in zip(periods, rng.normal(0, 0.1, len(periods)))},
              "e_i": {str(p): v for p, v in zip(periods, rng.normal(0, 0.2, len(periods)))}}
    truth = sol.solve(None, span, Scenario(set=shocks, unanticipated=list(shocks)))
    data = Dataset.from_frame(truth.frame.loc[span[0]:span[1]].copy())
    # observing output, potential output, inflation and the rate pins every state: the filter must give the truth back
    system = sol.system(["l_y", "l_y_bar", "pi", "i"], {s: 1.0 for s in shocks})
    smoothed = system.filter(data)
    for v in ("y_gap", "g_bar", "r", "dl_y", "l_y"):
        # 1e-4: without statsmodels, the native filter approximates the diffuse start by a large variance (about 1e-6 off)
        np.testing.assert_allclose(smoothed[v].iloc[2:], data[v].iloc[2:], atol=1e-4, err_msg=v)
    assert np.isfinite(system.loglike(data))
    # estimating a parameter of the trending model from the level of output, inflation and the rate
    measurement = __import__("pymetrix").MeasurementMap.direct(["l_y", "pi", "i"])
    fit = model.estimate(data, measurement, estimate=["a1", "sd_e_y"], shock_sd={"e_y": 0.5, "e_pi": 0.3, "e_g": 0.1, "e_i": 0.2},
                         guess={"l_y": 460, "l_y_bar": 460}, growth=True)
    est = dict(zip(fit.params.names, fit.params.values))
    se = dict(zip(fit.params.names, fit.params.se))
    assert abs(est["a1"] - 0.7) < 3 * se["a1"] and abs(est["sd_e_y"] - 0.5) < 3 * se["sd_e_y"]
