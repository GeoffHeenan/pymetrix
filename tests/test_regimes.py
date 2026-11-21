"""Known-date regimes: parameters and equations that change at given periods (pymetrix.models.regimes).

Checked independently:
- regimes equal to the base model, or one regime in force throughout, give exactly the results without regimes: the
  path (against the model solved at those values), the likelihood, the estimates and the forecast;
- a forecast through a path of known regimes (a new policy response, a temporary peg that ends, a new inflation target
  with another steady state, an alternative policy rule that drops the policy shock) is the nonlinear perfect-foresight
  solution (stacked Newton) of the same model written with the regimes' parameters as exogenous series, to 1e-8;
- a surprise regime (expected=False) is the known path without it up to its start, then the known path with it from
  the state reached;
- the regime filter's likelihood and filtered states are statsmodels' Kalman filter given the time-varying transition
  matrices built by hand from each regime's linearisation (backwards from the last regime's own solution);
- estimation recovers a regime's policy response (phi_pi@post) and the base one from data simulated with the change,
  within three standard errors; the Estimate and Forecast dialogs with regimes as text give the same results as code;
- refusals: regimes out of order or overlapping, unknown parameters and equations, regimes starting after the span or
  outside the sample, text that does not parse."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, Regime, parse_regimes, regime_filter, templates
from pymetrix.errors import SpecificationError
from pymetrix.metadata import Role
from pymetrix.models import MeasurementMap, Model, ModelError, PerfectForesight, Scenario
from pymetrix.models.linear import linearize
from pymetrix.models.regimes import regime_loglike, regime_model

COLS = ["y", "pi", "i"]
CAL = dict(sigma=1.0, beta=0.99, kappa=0.1, rho=0.8, phi_pi=1.5, phi_y=0.5, pistar=0.0)
ROLES = {s: Role.SHOCK for s in ("ey", "epi", "ei")}
RULE = "rho*i[-1] + (1 - rho)*(pistar + phi_pi*(pi - pistar) + phi_y*y) + ei"
ALT = "0.5*i[-1] + 0.5*(2*pi + 0.2*y)"  # an alternative rule, without the policy shock


def _equations(policy, params):
    return [Equation("y ~ y[+1] - (1/sigma)*(i - pi[+1]) + ey", params=["sigma"], name="IS"),
            Equation("pi ~ beta*pi[+1] + kappa*y + epi", params=["beta", "kappa"], name="Phillips"),
            Equation(policy, params=params, name="policy")]


def _model():
    return Model(_equations(f"i ~ {RULE}", ["rho", "phi_pi", "phi_y", "pistar"]), params=CAL, roles=ROLES)


def _span(frame, first="2025Q1", last=None):
    return frame.loc[first:last, COLS].to_numpy(float)


def _solution(model=None):
    return (model or _model()).linearize(shocks=["ey", "epi", "ei"]).solve()


def test_regimes_equal_to_the_base_or_one_regime_throughout_give_the_existing_results():
    nk = templates.new_keynesian()
    sol = nk.solve()
    span, sc = ("2025Q1", "2030Q4"), Scenario(set={"ey": {"2025Q1": -1.0}, "ei": {"2026Q1": 0.5}})
    plain = sol.solve(None, span, sc)
    same = sol.solve(None, span, sc, regimes=[Regime("2025Q3", {"phi_pi": 1.5}), Regime("2026Q1", {"kappa": 0.1})])
    assert np.array_equal(_span(plain.frame), _span(same.frame))
    one = sol.solve(None, span, sc, regimes=[Regime("2024Q4", {"phi_pi": 2.0})])  # in force before the span
    other = nk.model.with_parameters({"phi_pi": 2.0}).linearize(nk.steady_state(), shocks=nk.shocks).solve()
    assert np.array_equal(_span(one.frame), _span(other.solve(None, span, sc).frame))
    earlier = sol.solve(None, span, sc, regimes="2020Q1: phi_pi=2")
    assert np.array_equal(_span(one.frame), _span(earlier.frame))
    first = sol.solve(None, span, sc, regimes="2025Q1: phi_pi=2")  # from the first period: the rules, from the base state
    assert np.allclose(_span(one.frame), _span(first.frame), rtol=0, atol=1e-12)


def test_regimes_equal_to_the_base_give_the_existing_likelihood_estimates_and_forecast():
    nk = templates.new_keynesian()
    data = _nk_data(nk, 100, 5, [])
    system = nk.system()
    y, index = system.observations(data)
    assert regime_loglike(nk.solve(), system, [Regime("2010Q1", {"phi_pi": 1.5})], y, index) == system.loglike(data)
    # known-date regimes keep the numerical gradient (the analytic one does not cover them), so compare like with like
    plain = nk.estimate(data, estimate=["kappa"], gradient="numerical")
    same = nk.estimate(data, estimate=["kappa"], regimes=[Regime("2010Q1", {"phi_pi": 1.5})])
    assert same.params.values[0] == pytest.approx(nk.estimate(data, estimate=["kappa"]).params.values[0], rel=1e-6)
    assert same.params.values[0] == plain.params.values[0] and same.stats["loglike"] == plain.stats["loglike"]
    assert np.array_equal(same.smoothed.frame[COLS].to_numpy(), plain.smoothed.frame[COLS].to_numpy())
    a, b = plain.forecast(("2020Q1", "2022Q4")), same.forecast(("2020Q1", "2022Q4"))
    assert np.array_equal(_span(a.frame, "2020Q1"), _span(b.frame, "2020Q1"))


def _perfect_foresight_data(regime_series, first_data="2024Q1", last="2079Q4"):
    idx = pd.period_range(first_data, last, freq="Q", name="time")
    f = pd.DataFrame(0.0, index=idx, columns=[*COLS, "ey", "epi", "ei", "rho_t", "phi_t", "pis_t", "alt"])
    f["rho_t"], f["phi_t"], f["pis_t"] = CAL["rho"], CAL["phi_pi"], CAL["pistar"]
    for column, when, value in regime_series:
        f.loc[when[0]:when[1], column] = value
    f.loc["2024Q4", COLS] = [0.3, 0.2, 0.4]
    f.loc["2025Q1":, COLS] = np.nan
    f.loc["2025Q2", "ey"], f.loc["2027Q1", "ei"], f.loc["2025Q4", "epi"] = -1.0, 0.3, 0.2
    return Dataset.from_frame(f)


def test_a_known_regime_path_is_the_perfect_foresight_solution():
    """The same linear model with each regime's parameters as exogenous series (and the alternative rule switched in by
    an indicator series), solved by stacked Newton with the last regime's steady state as the terminal condition."""
    switched = Model(_equations(f"i ~ (1 - alt)*(rho_t*i[-1] + (1 - rho_t)*(pis_t + phi_t*(pi - pis_t) + phi_y*y) + ei) + alt*({ALT})",
                                ["phi_y"]), params={k: CAL[k] for k in ("sigma", "beta", "kappa", "phi_y")}, roles=ROLES)
    regimes = [Regime("2025Q3", {"phi_pi": 2.0}, name="hawk"),
               Regime("2026Q1", {"rho": 1.0, "phi_pi": 0.0}, end="2026Q4", name="peg"),  # then hawk again
               Regime("2028Q1", {"pistar": 0.5}, name="target"),  # another steady state
               Regime("2031Q1", equations={"policy": f"i = {ALT}"}, name="alternative")]
    series = [("phi_t", ("2025Q3", None), 2.0), ("rho_t", ("2026Q1", "2026Q4"), 1.0), ("phi_t", ("2026Q1", "2026Q4"), 0.0),
              ("pis_t", ("2028Q1", None), 0.5), ("alt", ("2031Q1", None), 1.0)]
    data = _perfect_foresight_data(series)
    nonlinear = switched.solve(data, ("2025Q1", "2079Q4"), solver=PerfectForesight(terminal="steady_state"))
    path = _solution().solve(data, ("2025Q1", "2079Q4"), regimes=regimes)
    assert np.max(np.abs(_span(nonlinear.frame, "2025Q1", "2045Q4") - _span(path.frame, "2025Q1", "2045Q4"))) < 1e-8
    spells = path.provenance.method["regimes"]
    assert [s["regime"] for s in spells] == ["base", "hawk", "peg", "hawk", "target", "alternative"]
    assert spells[2] == {"regime": "peg", "from": "2026Q1", "to": "2026Q4"}
    # the same through Model.solve with the first-order solver, and with the regimes written as text
    from pymetrix.models import FirstOrder

    text = _solution().solve(data, ("2025Q1", "2079Q4"), regimes="2025Q3: phi_pi=2; 2026Q1..2026Q4: rho=1, phi_pi=0; 2028Q1: pistar=0.5")
    by_model = _model().solve(data, ("2025Q1", "2079Q4"), solver=FirstOrder(), regimes=regimes[:3])
    assert np.max(np.abs(_span(text.frame, "2025Q1", "2030Q4") - _span(by_model.frame, "2025Q1", "2030Q4"))) < 1e-12
    assert np.max(np.abs(_span(text.frame, "2025Q1", "2030Q4") - _span(path.frame, "2025Q1", "2030Q4"))) > 1e-3  # the rule matters


def test_a_surprise_regime_is_the_path_without_it_until_it_starts_then_the_path_with_it():
    sol = _solution()
    data = _perfect_foresight_data([])
    known = [Regime("2026Q1", {"phi_pi": 2.0})]
    surprise = sol.solve(data, ("2025Q1", "2035Q4"), regimes=[Regime("2026Q1", {"phi_pi": 2.0}, expected=False)])
    without = sol.solve(data, ("2025Q1", "2035Q4"))
    assert np.allclose(_span(surprise.frame, "2025Q1", "2025Q4"), _span(without.frame, "2025Q1", "2025Q4"), atol=1e-12)
    frame = data.frame.copy()
    frame.loc["2025Q1":"2025Q4", COLS] = surprise.frame.loc["2025Q1":"2025Q4", COLS]  # where the surprise finds the economy
    restart = Dataset.from_frame(frame)
    after = sol.solve(restart, ("2026Q1", "2035Q4"), regimes=known)
    assert np.allclose(_span(surprise.frame, "2026Q1", "2035Q4"), _span(after.frame, "2026Q1", "2035Q4"), atol=1e-12)
    foreseen = sol.solve(data, ("2025Q1", "2035Q4"), regimes=known)
    assert not np.allclose(_span(foreseen.frame, "2025Q1", "2025Q4"), _span(without.frame, "2025Q1", "2025Q4"), atol=1e-6)


def _hand_rules(model, ss, shocks, keys_by_period, final_params):
    """Time-varying rules built here: each period's regime linearised at the base steady state, and P(t), k(t), Q(t)
    backwards from the last regime's own stable solution."""
    def lin(params):
        return linearize(model.with_parameters(params), ss, shocks, constant=True)

    last = lin(final_params)
    P = last.solve().P
    c = last.c if last.c is not None else np.zeros(len(P))
    k = -np.linalg.solve(last.A0 @ P + last.A1 + last.A0, c)
    out = []
    for params in reversed(keys_by_period):
        m = lin(params)
        M = m.A0 @ P + m.A1
        c = m.c if m.c is not None else np.zeros(len(P))
        P, k, Q = -np.linalg.solve(M, m.A2), -np.linalg.solve(M, m.A0 @ k + c), -np.linalg.solve(M, m.B)
        out.append((P, k, Q))
    return out[::-1]


def test_the_regime_filter_is_statsmodels_kalman_filter_with_the_time_varying_matrices():
    from statsmodels.tsa.statespace.mlemodel import MLEModel

    model = _model()
    shocks = ("ey", "epi", "ei")
    sol = _solution(model)
    rng = np.random.default_rng(4)
    idx = pd.period_range("2000Q1", periods=60, freq="Q", name="time")
    frame = pd.DataFrame(rng.normal(0, 0.5, (60, 3)), index=idx, columns=COLS)
    frame.iloc[7, 1] = np.nan  # a missing observation
    data = Dataset.from_frame(frame)
    measurement = MeasurementMap.direct(COLS, {"y": 0.1, "pi": 0.1, "i": 0.1})
    sd = {"ey": 0.5, "epi": 0.3, "ei": 0.2}
    regimes = [Regime("2004Q1", {"phi_pi": 2.0}), Regime("2008Q1", {"rho": 1.0, "phi_pi": 0.0}, end="2008Q4"),
               Regime("2010Q1", {"pistar": 0.5})]
    result = regime_filter(model, data, regimes, measurement, sd, shocks=shocks)
    params = [{}] * 16 + [{"phi_pi": 2.0}] * 16 + [{"phi_pi": 0.0, "rho": 1.0}] * 4 + [{"phi_pi": 2.0}] * 4 + [{"phi_pi": 2.0, "pistar": 0.5}] * 20
    rules = _hand_rules(model, sol.model.steady, shocks, params, {"phi_pi": 2.0, "pistar": 0.5})
    system = sol.system(measurement, sd)
    n, ns = 60, system.T.shape[0]
    T, c, R = np.repeat(system.T[None], n, 0), np.zeros((n, ns)), np.repeat(system.R[None], n, 0)
    for t, (P, k, Q) in enumerate(rules):
        T[t, :3, :3], c[t, :3], R[t, :3] = P, k, Q
    a0, P0, _ = system.start()
    y = frame.to_numpy() - system.d
    sm = MLEModel(y, k_states=ns, k_posdef=3)
    sm.ssm["design"], sm.ssm["obs_cov"], sm.ssm["state_cov"] = system.Z, system.H, system.Q
    sm.ssm["transition"] = np.moveaxis(np.concatenate([T[1:], T[-1:]]), 0, 2)  # statsmodels: period t to t+1
    sm.ssm["state_intercept"] = np.concatenate([c[1:], c[-1:]]).T
    sm.ssm["selection"] = np.moveaxis(np.concatenate([R[1:], R[-1:]]), 0, 2)
    sm.ssm.initialize_known(c[0] + T[0] @ a0, T[0] @ P0 @ T[0].T + R[0] @ system.Q @ R[0].T)  # the first period's state
    out = sm.filter([])
    assert abs(out.llf - result.loglike) < 1e-8
    assert np.max(np.abs(out.filtered_state[:3].T + 0.0 - (result.filtered.to_numpy() - 0.0))) < 1e-8
    assert list(result.regime.loc[["2003Q4", "2004Q1", "2008Q2", "2009Q1", "2010Q1"]]) == ["base", "2004Q1", "2008Q1", "2004Q1", "2010Q1"]
    assert {"ey", "epi", "ei", "y_fitted"} <= set(result.smoothed.frame.columns)
    assert "2010Q1" in str(result.summary())
    # with no regime change in the sample, the ordinary Kalman filter and smoother
    plain = regime_filter(model, data, None, measurement, sd, shocks=shocks)
    assert abs(plain.loglike - system.loglike(data)) < 1e-7
    assert np.max(np.abs(plain.smoothed.frame[COLS].to_numpy() - system.filter(data).frame[COLS].to_numpy())) < 1e-8


def _nk_data(nk, n, seed, regimes):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1995Q1", periods=n, freq="Q", name="time")
    shocks = pd.DataFrame(rng.normal(size=(n, 3)) * [nk.shock_sd[s] for s in nk.shocks], index=idx, columns=list(nk.shocks))
    sc = Scenario(set={s: shocks[s] for s in shocks}, unanticipated=list(nk.shocks))
    path = nk.solve().solve(None, (str(idx[0]), str(idx[-1])), sc, regimes=regimes)
    return nk.measurement.apply(Dataset.from_frame(path.frame.loc[idx, COLS]))


def test_estimation_recovers_a_regimes_parameter_and_the_dialogs_match_code():
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import EstimateModel, ForecastModel

    nk = templates.new_keynesian()
    data = _nk_data(nk, 160, 3, [Regime("2015Q1", {"phi_pi": 2.5}, name="post")])
    fit = nk.estimate(data, estimate=["phi_pi", "phi_pi@post"], regimes=[Regime("2015Q1", {"phi_pi": 1.5}, name="post")])
    values, se = dict(zip(fit.params.names, fit.params.values)), dict(zip(fit.params.names, fit.params.se))
    assert abs(values["phi_pi"] - 1.5) < 3 * se["phi_pi"] and abs(values["phi_pi@post"] - 2.5) < 3 * se["phi_pi@post"]
    assert values["phi_pi@post"] - values["phi_pi"] > 0.4
    assert fit.regimes[0].params["phi_pi"] == values["phi_pi@post"]
    assert fit.provenance.method["regimes"][-1] == {"regime": "post", "from": "2015Q1", "to": "2034Q4"}
    assert "Regimes: post" in str(fit.summary())
    plain = nk.estimate(data, estimate=["phi_pi"])
    assert fit.stats["loglike"] > plain.stats["loglike"]

    s = Session()
    s.objects.update({"nk": nk, "data": data})
    s.run(EstimateModel("nk", estimate="phi_pi, phi_pi@2015Q1", regimes="2015Q1: phi_pi=1.5", bounds="phi_pi@2015Q1=(1.01, 5)",
                        name="fit"))
    in_code = nk.estimate(data, estimate=["phi_pi", "phi_pi@2015Q1"], regimes=[Regime("2015Q1", {"phi_pi": 1.5})],
                          bounds={"phi_pi@2015Q1": (1.01, 5)})
    assert np.array_equal(s["fit"].params.values, in_code.params.values)
    assert np.allclose(s["fit"].params.values, fit.params.values, rtol=0, atol=1e-4)  # the same mode, other bounds
    assert "regimes='2015Q1: phi_pi=1.5'" in s.script()
    s.run(ForecastModel("fit", "2035Q1", "2037Q4", regimes="2035Q2..2035Q4: rho=1, phi_pi=0", name="fc"))
    in_code = s["fit"].forecast(("2035Q1", "2037Q4"), regimes=[Regime("2035Q2", {"rho": 1.0, "phi_pi": 0.0}, end="2035Q4")])
    assert np.array_equal(_span(s["fc"].frame, "2035Q1"), _span(in_code.frame, "2035Q1"))
    assert "regimes='2035Q2..2035Q4: rho=1, phi_pi=0'" in s.script()
    assert np.ptp(s["fc"].frame.loc["2035Q2":"2035Q4", "i"]) < 1e-12  # the peg holds the rate
    assert ForecastModel("fit", "2035Q1", "2037Q4", regimes="2035Q1 phi_pi=1").validate(s)
    # a forecast's draws through the regimes
    drawn = s["fit"].forecast(("2035Q1", "2037Q4"), regimes="2035Q2..2035Q4: rho=1, phi_pi=0", draws=40, seed=2)
    assert drawn.simulation is not None and np.ptp(drawn.quantiles([0.5])["i_q50"].loc["2035Q2":"2035Q4"]) < 0.2


def test_an_alternative_equation_replaces_the_rule_by_name():
    model = _model()
    rule = Model(_equations(f"i = {ALT}", []), params=CAL, roles=ROLES)
    replaced = regime_model(model, {}, {"policy": f"i = {ALT}"})
    assert replaced.endogenous == model.endogenous
    sol = _solution()
    data = _perfect_foresight_data([])
    a = sol.solve(data, ("2025Q1", "2030Q4"), regimes=[Regime("2020Q1", equations={"policy": f"i = {ALT}"})])
    b = rule.linearize(shocks=["ey", "epi"]).solve()
    from pymetrix.models.plans import solve_plan

    bb = solve_plan(b, data, ("2025Q1", "2030Q4"))
    assert np.allclose(_span(a.frame, "2025Q1", "2030Q4"), _span(bb.frame, "2025Q1", "2030Q4"), atol=1e-12)
    with_param = regime_model(model, {"g": 0.7}, {"policy": "i = g*i[-1] + 2*(1 - g)*pi"})
    assert "g" in with_param.parameters.names


def test_refusals():
    sol = _solution()
    data = _perfect_foresight_data([])
    span = ("2025Q1", "2030Q4")
    with pytest.raises(SpecificationError, match="order of their starts"):
        sol.solve(data, span, regimes=[Regime("2026Q1", {"phi_pi": 2}), Regime("2025Q3", {"phi_pi": 1})])
    with pytest.raises(SpecificationError, match="before the temporary regime"):
        sol.solve(data, span, regimes=[Regime("2026Q1", {"phi_pi": 2}, end="2026Q4"), Regime("2026Q3", {"rho": 0.5})])
    with pytest.raises(ModelError, match="not parameters of the model"):
        sol.solve(data, span, regimes=[Regime("2026Q1", {"phi": 2})])
    with pytest.raises(ModelError, match="not equations of the model"):
        sol.solve(data, span, regimes=[Regime("2026Q1", equations={"taylor": "i = pi"})])
    with pytest.raises(ModelError, match="neither variables"):
        sol.solve(data, span, regimes=[Regime("2026Q1", equations={"policy": "i = pi + zz"})])
    with pytest.raises(ModelError, match="determines"):
        sol.solve(data, span, regimes=[Regime("2026Q1", equations={"policy": "pi = i"})])
    with pytest.raises(ModelError, match="after the span ends"):
        sol.solve(data, span, regimes="2031Q1: phi_pi=2")
    with pytest.raises(ModelError, match="no unique stable solution"):
        sol.solve(data, span, regimes="2026Q1: phi_pi=0.5")  # the last regime holds for good: indeterminate
    with pytest.raises(SpecificationError, match="period: name=value"):
        parse_regimes("2026Q1 phi_pi=2")
    with pytest.raises(SpecificationError, match="a number"):
        parse_regimes("2026Q1: phi_pi=high")
    with pytest.raises(SpecificationError, match="changes nothing"):
        Regime("2026Q1")
    with pytest.raises(SpecificationError, match="regimes are Regime"):
        sol.solve(data, span, regimes=[{"start": "2026Q1"}])
    nk = templates.new_keynesian()
    small = _nk_data(nk, 40, 1, [])
    for text in ("1995Q1: phi_pi=2", "2010Q1: phi_pi=2"):  # at the first period, after the last
        with pytest.raises(ModelError, match="outside the data"):
            nk.estimate(small, estimate=["kappa"], regimes=text)
    with pytest.raises(ModelError, match="not parameters, shock sizes"):
        nk.estimate(small, estimate=["kappa@post"], regimes="2000Q1: phi_pi=2")
    from pymetrix.models import Newton

    with pytest.raises(SpecificationError, match="first-order solver"):
        _model().solve(data, span, solver=Newton(), regimes="2026Q1: phi_pi=2")
    assert parse_regimes("2009Q1: phi_pi=1.2; 2025Q1..2026Q4: rho=1, phi_pi=0") == (
        Regime("2009Q1", {"phi_pi": 1.2}), Regime("2025Q1", {"rho": 1.0, "phi_pi": 0.0}, end="2026Q4"))
