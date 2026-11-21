"""Known-date regimes, OccBin and correlated shocks together (pymetrix.models.regimes, occbin, occbin_filter,
perfect_foresight, decompose, stochastic).

Checked independently:
- without regimes or correlations nothing changes: regimes equal to the base model give, bit for bit, the OccBin path,
  the OccBin filter, the perfect-foresight path and the decompositions without them;
- perfect foresight with regimes (the stacked system's equations switching at their dates) is the time-varying
  first-order solution on a model whose equations are linear (1e-8), and, for a nonlinear model (the growth model
  with a new discount factor for good and a temporary depreciation rate), the same model solved by stacked Newton
  with the changing parameters written as series in the data (a step function), 1e-8;
- OccBin with regimes is the nonlinear perfect-foresight solution of the model with its max, with the regimes'
  parameters as series in the data (anticipated shocks and repeated surprises, 1e-8), and perfect foresight with the
  same regimes gives the same path;
- the OccBin filter with correlated shocks (Q off its diagonal) is the linear Kalman filter with the same Q when
  nothing binds (likelihood, smoothed states and shocks, 1e-8); with regimes and correlations on a sample where the
  constraint binds, its likelihood and filtered states are statsmodels' Kalman filter given time-varying matrices
  built here from each period's regime and constraint branch (linearised by hand, backwards from the last regime's
  own solution), 1e-8; estimation with the OccBin filter and correlations, where nothing binds, is the linear
  estimation with correlations; the decompositions add up to the smoothed path (1e-10), orthogonalised or not;
- OccBin forecasts draw correlated shocks: the sample covariance of the drawn shocks is the covariance given;
  OccBin forecasts with regimes, where nothing binds, are the linear forecasts with the same regimes;
- posterior forecasts of a model estimated with regimes: each draw is the regime forecast worked out again at that
  draw's parameters through the template's own regime filter and solution (1e-10); the Forecast dialog takes
  regimes for a posterior and gives what its recorded code gives;
- regime decompositions add up to the smoothed path (1e-10), equal ``LinearSolution.decompose`` with the regimes on
  that path, and the linear decomposition when the regimes equal the base; a regime's moments are those of the model
  at the regime's values on its own."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, Model, Regime, occbin, occbin_filter, regime_filter, templates
from pymetrix.estimation import priors
from pymetrix.metadata import Role
from pymetrix.models import MeasurementMap, ModelError, PerfectForesight, Scenario
from pymetrix.models.linear import linearize
from pymetrix.models.occbin import _Piecewise
from pymetrix.models.occbin_filter import occbin_forecast, piecewise_filter

COLS = ["y", "pi", "i"]
SH = ("ey", "epi", "ei")
NOISY = MeasurementMap.direct(COLS, {"y": 0.02, "pi": 0.02, "i": 0.02})
CORR = {("ey", "epi"): 0.4, ("epi", "ei"): -0.3}


# ---------------------------------------------------------------------------------------------
# models


def _nk_linear(policy="i ~ rho*i[-1] + (1 - rho)*(pistar + phi_pi*(pi - pistar) + phi_y*y) + ei"):
    eqs = [Equation("y ~ y[+1] - (1/sigma)*(i - pi[+1]) + ey", params=["sigma"], name="IS"),
           Equation("pi ~ beta*pi[+1] + kappa*y + epi", params=["beta", "kappa"], name="Phillips"),
           Equation(policy, params=["rho", "phi_pi", "phi_y", "pistar"], name="policy")]
    return Model(eqs, params=dict(sigma=1.0, beta=0.99, kappa=0.1, rho=0.8, phi_pi=1.5, phi_y=0.5, pistar=0.0),
                 roles={s: Role.SHOCK for s in SH})


def _zlb(step=False):
    """A New Keynesian model with a natural rate and a zero lower bound; ``step``: phi and rbar as series in the data."""
    shadow = Equation("ishadow = rn + phi_t*pi + 0.5*y") if step else Equation("ishadow ~ rn + phi*pi + 0.5*y", params=["phi"])
    rn = Equation("rn = 0.2*rbar_t + 0.8*rn[-1] + ez") if step else Equation("rn ~ (1-rho)*rbar + rho*rn[-1] + ez", params=["rho", "rbar"])
    params = dict(sig=1.0, bet=0.99, kap=0.1) if step else dict(sig=1.0, bet=0.99, kap=0.1, phi=1.5, rbar=1.0, rho=0.8)
    eqs = [Equation("y ~ y[+1] - sig*(i - pi[+1] - rn) + ey", params=["sig"]),
           Equation("pi ~ bet*pi[+1] + kap*y + epi", params=["bet", "kap"]), shadow,
           Equation("i = max(ishadow, 0)", name="lower bound"), rn]
    return Model(eqs, params=params, roles={s: Role.SHOCK for s in ("ey", "epi", "ez")})


def _growth(step=False):
    """The stochastic growth model, nonlinear; ``step``: beta and delta as series in the data."""
    b, d = ("beta_t", "delta_t") if step else ("beta", "delta")
    own = [] if step else ["beta", "delta"]
    eqs = [Equation(f"c ~ c[+1] / ({b}*(1 + alpha*k^(alpha - 1) - {d}))", params=["alpha", *own], name="Euler"),
           Equation("y ~ exp(e)*k[-1]^alpha", params=["alpha"], name="production"),
           Equation(f"k = y + (1 - {d})*k[-1] - c", name="capital") if step else
           Equation("k ~ y + (1 - delta)*k[-1] - c", params=["delta"], name="capital")]
    params = {"alpha": 0.33} if step else {"alpha": 0.33, "beta": 0.99, "delta": 0.025}
    return Model(eqs, params=params, roles={"e": Role.SHOCK})


def _quarters(n, first="2000Q1"):
    return pd.period_range(first, periods=n, freq="Q", name="time")


# ---------------------------------------------------------------------------------------------
# nothing changes without regimes or correlations


def test_regimes_equal_to_the_base_and_no_correlations_change_nothing():
    m = _zlb()
    shocks = {"ez": {1: -2.0}, "ey": {3: -0.5}}
    plain = occbin(m, shocks, periods=30, shock_names=["ey", "epi", "ez"])
    same = occbin(m, shocks, periods=30, shock_names=["ey", "epi", "ez"], regimes=[Regime(5, {"phi": 1.5})])
    assert np.array_equal(plain.frame.to_numpy(), same.frame.to_numpy())
    assert np.array_equal(plain.linear.to_numpy(), same.linear.to_numpy())

    zlb = templates.new_keynesian(lower_bound=2.0)
    data = _occbin_data(zlb, 60, 2, [])
    a = occbin_filter(zlb, data, NOISY)
    b = occbin_filter(zlb, data, NOISY, regimes=[Regime("1990Q1", {"phi_pi": 1.5})], shock_corr={("ey", "epi"): 0.0})
    assert a.loglike == b.loglike
    assert np.array_equal(a.smoothed.frame.to_numpy(), b.smoothed.frame.to_numpy())
    da, db = a.decompose(), b.decompose()
    assert all(np.array_equal(da.contributions[v].to_numpy(), db.contributions[v].to_numpy()) for v in da.contributions)

    data = _pf_data(_quarters(60), {})
    pf = _nk_linear().solve(data, ("2000Q2", "2014Q4"), solver=PerfectForesight(terminal="steady_state"))
    pf_same = _nk_linear().solve(data, ("2000Q2", "2014Q4"), solver=PerfectForesight(terminal="steady_state"),
                                 regimes=[Regime("2003Q1", {"phi_pi": 1.5})])
    assert np.array_equal(pf.frame[COLS].to_numpy(), pf_same.frame[COLS].to_numpy())


# ---------------------------------------------------------------------------------------------
# perfect foresight


def _pf_data(index, columns):
    frame = pd.DataFrame(0.0, index=index, columns=[*COLS, *SH])
    frame.loc[index[0], COLS] = [0.3, 0.2, 0.4]
    frame.loc[index[1]:, COLS] = np.nan
    frame.loc[index[2], "ey"], frame.loc[index[8], "ei"] = -1.0, 0.3
    for name, values in columns.items():
        frame[name] = values
    return Dataset.from_frame(frame)


def test_perfect_foresight_with_regimes_is_the_time_varying_first_order_solution():
    regimes = [Regime("2001Q3", {"phi_pi": 2.0}, name="hawk"),
               Regime("2002Q1", {"rho": 1.0, "phi_pi": 0.0}, end="2002Q4", name="peg"),
               Regime("2004Q1", {"pistar": 0.5}, name="target"),
               Regime("2007Q1", equations={"policy": "i = 0.5*i[-1] + 0.5*(2*pi + 0.2*y)"}, name="alternative")]
    index = _quarters(200)
    data = _pf_data(index, {})
    span = ("2000Q2", str(index[-1]))
    pf = _nk_linear().solve(data, span, solver=PerfectForesight(terminal="steady_state"), regimes=regimes)
    first_order = _nk_linear().linearize(shocks=list(SH)).solve().solve(data, span, regimes=regimes)
    gap = pf.frame.loc["2000Q2":"2030Q4", COLS].to_numpy() - first_order.frame.loc["2000Q2":"2030Q4", COLS].to_numpy()
    assert np.max(np.abs(gap)) < 1e-8
    assert [s["regime"] for s in pf.provenance.method["regimes"]] == ["base", "hawk", "peg", "hawk", "target", "alternative"]
    assert np.ptp(pf.frame.loc["2002Q1":"2002Q4", "i"]) < 1e-10  # the peg holds the rate
    # the same through Model.solve's surprises (an unanticipated shock): solved again from it, with the regimes
    data = _pf_data(index, {"ey": 0.0, "ei": 0.0})  # the shocks all in the scenario
    sc = Scenario(set={"ei": {"2003Q2": 0.2}}, unanticipated=["ei"])
    news = _nk_linear().solve(data, span, sc, solver=PerfectForesight(terminal="steady_state"), regimes=regimes)
    tv = _nk_linear().linearize(shocks=list(SH)).solve().solve(data, span, sc, regimes=regimes)
    assert np.max(np.abs(news.frame.loc["2000Q2":"2030Q4", COLS].to_numpy() - tv.frame.loc["2000Q2":"2030Q4", COLS].to_numpy())) < 1e-8


def test_perfect_foresight_with_regimes_of_a_nonlinear_model_is_the_model_with_the_parameters_as_data():
    index = _quarters(240)
    guess = {"k": 30.0, "y": 3.0, "c": 2.3}
    ss = _growth().steady_state(guess=guess)
    beta = pd.Series(0.99, index=index)
    beta.loc["2005Q1":] = 0.985  # for good: another steady state
    delta = pd.Series(0.025, index=index)
    delta.loc["2002Q1":"2003Q4"] = 0.04  # for a while
    frame = pd.DataFrame({"c": ss["c"], "k": ss["k"], "y": ss["y"], "e": 0.0, "beta_t": beta, "delta_t": delta}, index=index)
    frame.loc["2000Q3", "e"] = 0.01  # the span's values of c, k and y are the starting values (the old steady state)
    data = Dataset.from_frame(frame)
    span = ("2000Q2", str(index[-1]))
    by_data = _growth(step=True).solve(data, span, solver=PerfectForesight(terminal="steady_state"))
    regimes = [Regime("2002Q1", {"delta": 0.04}, end="2003Q4", name="wear"), Regime("2005Q1", {"beta": 0.985}, name="impatient")]
    by_regimes = _growth().solve(data, span, solver=PerfectForesight(terminal="steady_state"), regimes=regimes)
    gap = by_regimes.frame.loc[span[0]:, ["c", "k", "y"]].to_numpy() - by_data.frame.loc[span[0]:, ["c", "k", "y"]].to_numpy()
    assert np.max(np.abs(gap)) < 1e-8
    new_ss = _growth().with_parameters({"beta": 0.985}).steady_state(guess=guess)
    assert abs(by_regimes.frame["k"].iloc[-1] - new_ss["k"]) < 1e-3 and abs(new_ss["k"] - ss["k"]) > 1  # a new steady state
    # one regime throughout the span: that regime's model
    one = _growth().solve(data, span, solver=PerfectForesight(terminal="steady_state"), regimes="1999Q1: delta=0.03")
    other = _growth().with_parameters({"delta": 0.03}).solve(data, span, solver=PerfectForesight(terminal="steady_state"))
    assert np.array_equal(one.frame[["c", "k", "y"]].to_numpy(), other.frame[["c", "k", "y"]].to_numpy())
    # surprise regimes (expected=False) with perfect foresight: tests/test_regime_gaps.py


def test_the_solve_dialog_takes_regimes_with_perfect_foresight():
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import DefineEquation, SolveModel

    s = Session()
    s.run(DefineEquation("y ~ y[+1] - s*(i - pi[+1]) + ey", "s", name="eq_is"))
    s.run(DefineEquation("pi ~ b*pi[+1] + k*y + epi", "b, k", name="eq_pc"))
    s.run(DefineEquation("i ~ r*i[-1] + (1-r)*(phi*pi + 0.5*y) + ei", "r, phi", name="eq_tr"))
    s.objects["data"] = _pf_data(_quarters(80), {})
    cmd = SolveModel("eq_is, eq_pc, eq_tr", "2000Q2", "2019Q4", solver="perfect_foresight", solver_settings="terminal='steady_state'",
                     params="s=1, b=0.99, k=0.1, r=0.8, phi=1.5", regimes="2001Q1..2001Q4: r=1, phi=0; 2003Q1: phi=2")
    assert cmd.validate(s) == []
    s.run(cmd)
    assert "regimes='2001Q1..2001Q4: r=1, phi=0; 2003Q1: phi=2'" in s.script()
    model = Model([s["eq_is"], s["eq_pc"], s["eq_tr"]], params={"s": 1.0, "b": 0.99, "k": 0.1, "r": 0.8, "phi": 1.5})
    in_code = model.solve(s["data"], ("2000Q2", "2019Q4"), solver=PerfectForesight(terminal="steady_state"),
                          regimes="2001Q1..2001Q4: r=1, phi=0; 2003Q1: phi=2")
    assert np.array_equal(s["solution"].frame[COLS].to_numpy(), in_code.frame[COLS].to_numpy())
    assert np.ptp(in_code.frame.loc["2001Q1":"2001Q4", "i"]) < 1e-10


# ---------------------------------------------------------------------------------------------
# OccBin with regimes


def _zlb_by_data(shocks, T, scenario=None):
    """The model with its max solved by stacked Newton, phi and rbar as series in the data (the regimes' dates)."""
    ss = _zlb().steady_state()
    idx = _quarters(T + 121)
    frame = pd.DataFrame({v: ss[v] for v in _zlb().endogenous}, index=idx)
    for s in ("ey", "epi", "ez"):
        frame[s] = 0.0
    for name, path in shocks.items():
        for t, value in path.items():
            frame.loc[idx[t], name] = value
    frame["phi_t"], frame["rbar_t"] = 1.5, 1.0
    frame.loc[idx[3]:, "phi_t"] = 2.5
    frame.loc[idx[9]:idx[12], "phi_t"] = 0.5
    frame.loc[idx[20]:, "rbar_t"] = 0.6
    sol = _zlb(step=True).solve(Dataset.from_frame(frame), (idx[1], idx[-1]), scenario, solver=PerfectForesight(terminal="steady_state"))
    return sol.frame.iloc[1:T + 1].reset_index(drop=True), idx, frame


REGIMES = [Regime(3, {"phi": 2.5}, name="hawk"), Regime(9, {"phi": 0.5}, end=12, name="dove"), Regime(20, {"rbar": 0.6}, name="low")]


def test_occbin_with_regimes_is_perfect_foresight_with_the_max_and_the_regimes_dates():
    m, names = _zlb(), ["ey", "epi", "ez"]
    shocks = {"ez": {1: -2.0}, "ey": {3: -0.6, 10: -0.8}}
    known = occbin(m, shocks, periods=40, anticipated=True, shock_names=names, regimes=REGIMES, horizon=200)
    pf, idx, frame = _zlb_by_data(shocks, 40)
    for v in ("y", "pi", "i", "rn"):
        assert np.max(np.abs(known.frame[v].to_numpy() - pf[v].to_numpy())) < 1e-8, v
    binds = known.binding.iloc[:, 0]
    assert binds.sum() >= 4 and binds.iloc[2] and binds.iloc[3]  # binding across the hawk regime's start
    assert abs(known.frame["i"].iloc[-1] - (0.6 + 0.0)) < 0.05  # towards the low regime's steady state
    # perfect foresight with the same regimes (the model's own max), dated
    dated = [Regime(str(idx[3]), {"phi": 2.5}, name="hawk"), Regime(str(idx[9]), {"phi": 0.5}, end=str(idx[12]), name="dove"),
             Regime(str(idx[20]), {"rbar": 0.6}, name="low")]
    by_regimes = m.solve(Dataset.from_frame(frame.drop(columns=["phi_t", "rbar_t"])), (idx[1], idx[-1]), regimes=dated,
                         solver=PerfectForesight(terminal="steady_state"))
    assert np.max(np.abs(by_regimes.frame.iloc[1:41][["y", "pi", "i"]].to_numpy() - pf[["y", "pi", "i"]].to_numpy())) < 1e-8
    # repeated surprises: solved again in each period, the regimes known throughout
    surprises = occbin(m, shocks, periods=40, shock_names=names, regimes=REGIMES, horizon=200)
    sc = Scenario(set={"ez": {str(idx[1]): -2.0}, "ey": {str(idx[3]): -0.6, str(idx[10]): -0.8}}, unanticipated=["ez", "ey"])
    pf_s, _, _ = _zlb_by_data({}, 40, sc)
    for v in ("y", "pi", "i"):
        assert np.max(np.abs(surprises.frame[v].to_numpy() - pf_s[v].to_numpy())) < 1e-8, v
    # the path without the constraint follows the regimes too
    linear = m.linearize(shocks=names).solve().solve(None, (str(idx[1]), str(idx[40])), Scenario(
        set={"ez": {str(idx[1]): -2.0}, "ey": {str(idx[3]): -0.6, str(idx[10]): -0.8}}, unanticipated=["ez", "ey"]), regimes=dated)
    assert np.max(np.abs(surprises.linear["i"].to_numpy() - linear.frame.loc[idx[1]:idx[40], "i"].to_numpy())) < 1e-10


def test_occbin_refuses_regimes_it_cannot_take():
    m = _zlb()
    # regimes that remove or add a constraint, and surprise regimes, are taken now: tests/test_regime_gaps.py
    with pytest.raises(Exception, match="period number"):
        occbin(m, {"ez": {1: -2.0}}, periods=10, shock_names=["ey", "epi", "ez"], regimes="2001Q1: phi=2")


# ---------------------------------------------------------------------------------------------
# the OccBin filter with correlated shocks and regimes


def _occbin_data(zlb, n, seed, regimes, extra=None, noise=0.02):
    rng = np.random.default_rng(seed)
    shocks = {s: {t + 1: float(v) for t, v in enumerate(rng.normal(0, zlb.shock_sd[s], n))} for s in SH}
    for t, value in (extra or {}).items():
        shocks["ey"][t] += value
    path = occbin(zlb, shocks, periods=n, regimes=regimes)
    frame = path.frame[COLS].copy()
    frame.index = _quarters(n, "1980Q1")
    return Dataset.from_frame(frame + rng.normal(0, noise, frame.shape))


def test_the_occbin_filter_with_correlations_is_the_linear_filter_when_nothing_binds():
    zlb = templates.new_keynesian(lower_bound=2.0)
    data = _occbin_data(zlb, 80, 3, [])
    far = templates.new_keynesian(lower_bound=-100.0)
    result = occbin_filter(far, data, NOISY, shock_corr=CORR)
    system = far.solve().system(NOISY, far.shock_sd, shock_corr=CORR)
    assert not result.binding.to_numpy().any()
    assert abs(result.loglike - system.loglike(data)) < 1e-8
    linear = system.filter(data).frame
    assert np.max(np.abs(result.smoothed.frame[COLS].to_numpy() - linear[COLS].to_numpy())) < 1e-8
    # the smoothed shocks: those the linear path implies (the linear decomposition's)
    decomposition = far.solve().decompose(linear)
    shocks = result.smoothed.frame[list(SH)].to_numpy()[1:]
    assert np.max(np.abs(shocks - decomposition.shocks.to_numpy()[1:])) < 1e-8
    own = result.decompose(orthogonal=False)
    assert max(np.max(np.abs(own.contributions[v][list(SH)].to_numpy() - decomposition.contributions[v][list(SH)].to_numpy()))
               for v in COLS) < 1e-8
    orthogonal = result.decompose()
    assert "orthogonalised" in orthogonal.title and np.max(np.abs(orthogonal.residual.to_numpy())) < 1e-10
    assert max(np.max(np.abs(orthogonal.contributions[v]["binding constraints"].to_numpy())) for v in COLS) == 0.0


def _hand_rules(zlb, ss, run, n, regime_params, last_params):
    """Each period's rule built here: the regime of each period ahead and the branch of the constraint the filter
    expected, each written out as its own model (the rule, or the rate at its floor) and linearised at the reference
    steady state; backwards from the rules with the constraint slack, which come from the last regime's own solution."""
    base = dict(zlb.model.params)

    def model(params, binding):
        p = {**base, **params}
        policy = (Equation("i ~ i_floor + 0*ei", params=["i_floor"], name="policy") if binding else
                  Equation("i ~ rho*i[-1] + (1 - rho)*(phi_pi*pi + phi_y*y) + ei", params=["rho", "phi_pi", "phi_y"], name="policy"))
        eqs = [Equation("y ~ y[+1] - (1/sigma)*(i - pi[+1]) + ey", params=["sigma"], name="IS"),
               Equation("pi ~ beta*pi[+1] + kappa*y + epi", params=["beta", "kappa"], name="Phillips"), policy]
        used = {"sigma", "beta", "kappa", *(("i_floor",) if binding else ("rho", "phi_pi", "phi_y"))}
        return linearize(Model(eqs, params={k: v for k, v in p.items() if k in used}, roles={s: Role.SHOCK for s in SH}), ss, SH,
                         constant=True)

    cache = {}

    def lin(s, binding):
        key = (tuple(sorted(regime_params(s).items())), bool(binding))
        if key not in cache:
            cache[key] = model(regime_params(s), binding)
        return cache[key]

    last = model(last_params, False)
    P_final = last.solve().P
    d_final = -np.linalg.solve(last.A0 @ P_final + last.A1 + last.A0, last.c)

    def slack_from(s, change):  # the rule with the constraint slack from period s on
        P, k = P_final, d_final
        for u in range(max(s, change) - 1, s - 1, -1):
            m = lin(u, False)
            M = m.A0 @ P + m.A1
            P, k = -np.linalg.solve(M, m.A2), -np.linalg.solve(M, m.A0 @ k + m.c)
        return P, k

    out = []
    for t in range(n):
        expected = run["regimes"][t]
        P, k = slack_from(t + len(expected), n)
        for s in range(len(expected) - 1, -1, -1):
            m = lin(t + s, expected[s, 0])
            M = m.A0 @ P + m.A1
            P, k, Q = -np.linalg.solve(M, m.A2), -np.linalg.solve(M, m.A0 @ k + m.c), -np.linalg.solve(M, m.B)
        out.append((P, k, Q))
    return out


def test_the_occbin_filter_with_regimes_and_correlations_is_statsmodels_given_the_regimes():
    from statsmodels.tsa.statespace.kalman_filter import KalmanFilter

    zlb = templates.new_keynesian(lower_bound=2.0)
    n = 100
    data = _occbin_data(zlb, n, 1, [Regime(31, {"phi_pi": 2.0}, name="hawk"), Regime(61, {"i_floor": -0.4}, name="lower")],
                        {**{t: -3.0 for t in (40, 41, 42)}, **{t: -4.0 for t in (66, 67, 68)}})
    idx = data.frame.index
    regimes = [Regime(str(idx[30]), {"phi_pi": 2.0}, name="hawk"), Regime(str(idx[60]), {"i_floor": -0.4}, name="lower")]
    pw = _Piecewise(zlb.model, SH, {}, 40, regimes=regimes, first=idx[0], freq=idx.freq, length=n)
    pw.tie = True
    system = pw.solution.system(NOISY, zlb.shock_sd, shock_corr=CORR)
    assert np.any(system.Q - np.diag(np.diag(system.Q)))
    y, _ = system.observations(data)
    run = piecewise_filter(pw, system, y)
    binds = run["binds"][:, 0]
    assert binds[30:60].sum() >= 2 and binds[60:].sum() >= 2  # binding in both regimes

    def regime_params(s):
        return {**({"phi_pi": 2.0} if s >= 30 else {}), **({"i_floor": -0.4} if s >= 60 else {})}

    rules = _hand_rules(zlb, zlb.model.steady_state(), run, n, regime_params, {"phi_pi": 2.0, "i_floor": -0.4})
    ns = system.T.shape[0]
    T, c, R = np.repeat(system.T[None], n, 0), np.zeros((n, ns)), np.repeat(system.R[None], n, 0)
    for t, (P, k, Q) in enumerate(rules):
        T[t, :3, :3], c[t, :3], R[t, :3] = P, k, Q
    kf = KalmanFilter(k_endog=3, k_states=ns, k_posdef=3)
    kf.bind(np.asarray(y - system.d, dtype=float))
    Tm, cm, Rm = np.moveaxis(T, 0, -1), c.T, np.moveaxis(R, 0, -1)
    kf["design"], kf["obs_cov"], kf["state_cov"] = system.Z, system.H, system.Q
    kf["transition"] = np.concatenate([Tm[:, :, 1:], Tm[:, :, -1:]], axis=2)
    kf["state_intercept"] = np.concatenate([cm[:, 1:], cm[:, -1:]], axis=1)
    kf["selection"] = np.concatenate([Rm[:, :, 1:], Rm[:, :, -1:]], axis=2)
    a0, P0, _ = system.start()
    kf.initialize_known(c[0] + T[0] @ a0, T[0] @ P0 @ T[0].T + R[0] @ system.Q @ R[0].T)
    assert abs(kf.loglike() - run["loglike"]) < 1e-8
    assert np.max(np.abs(kf.filter().filtered_state.T - run["a_filt"])) < 1e-8
    # the public filter gives the same, and its decompositions add up to the smoothed path
    result = occbin_filter(zlb, data, NOISY, regimes=regimes, shock_corr=CORR)
    assert result.loglike == run["loglike"]
    for d in (result.decompose(), result.decompose(orthogonal=False)):
        assert np.max(np.abs(d.residual.to_numpy())) < 1e-10
    assert np.max(np.abs(result.decompose().contributions["i"]["binding constraints"].to_numpy())) > 1e-3
    iterated = occbin_filter(zlb, data, NOISY, regimes=regimes, shock_corr=CORR, smoother="iterated")
    assert np.max(np.abs(iterated.decompose().residual.to_numpy())) < 1e-10


def test_estimation_with_the_occbin_filter_and_correlations():
    zlb = templates.new_keynesian(lower_bound=2.0)
    data = _occbin_data(zlb, 80, 3, [])
    far = templates.new_keynesian(lower_bound=-100.0)
    plain = far.model.estimate(data, NOISY, shocks=SH, shock_sd=far.shock_sd, estimate=["rho", "corr_ey_epi"],
                               shock_corr={("ey", "epi"): 0.2}, cov="none")
    via_filter = far.model.estimate(data, NOISY, shocks=SH, shock_sd=far.shock_sd, estimate=["rho", "corr_ey_epi"],
                                    shock_corr={("ey", "epi"): 0.2}, cov="none", occbin=True)
    assert np.allclose(via_filter.params.values, plain.params.values, rtol=0, atol=1e-4)
    assert abs(via_filter.stats["loglike"] - plain.stats["loglike"]) < 1e-6
    assert via_filter.shock_corr and "OccBin" in via_filter.method
    # with regimes too: the likelihood at the estimates is the OccBin filter's there
    regimes = [Regime("1995Q1", {"phi_pi": 1.5}, name="hawk")]
    fit = zlb.model.estimate(data, NOISY, shocks=SH, shock_sd=zlb.shock_sd, estimate=["phi_pi@hawk"], shock_corr=CORR,
                             regimes=regimes, occbin=True, cov="none")
    again = occbin_filter(fit, data)
    assert abs(again.loglike - fit.stats["loglike"]) < 1e-8
    assert fit.regimes[0].params["phi_pi"] == fit.params.as_dict()["phi_pi@hawk"]


# ---------------------------------------------------------------------------------------------
# OccBin forecasts


def test_occbin_forecast_draws_are_correlated(monkeypatch):
    import sys

    ob = sys.modules["pymetrix.models.occbin"]

    far = templates.new_keynesian(lower_bound=-100.0)
    idx = _quarters(4, "2020Q1")
    smoothed = Dataset.from_frame(pd.DataFrame(0.0, index=idx, columns=COLS))
    drawn = []
    simulate = ob._simulate

    def keep(pw, x0, E, anticipated):
        drawn.append(E.copy())
        return simulate(pw, x0, E, anticipated)

    monkeypatch.setattr(ob, "_simulate", keep)
    sd = {"ey": 0.5, "epi": 0.3, "ei": 0.2}
    _, _, sims, _ = occbin_forecast(far.model, SH, {}, smoothed, ("2021Q1", "2023Q4"), draws=400, shock_sd=sd, seed=4,
                                    shock_corr=CORR)
    shocks = np.concatenate(drawn[1:])
    target = np.diag(list(sd.values())) @ np.array([[1, 0.4, 0], [0.4, 1, -0.3], [0, -0.3, 1]]) @ np.diag(list(sd.values()))
    assert np.max(np.abs(np.cov(shocks.T) / np.outer(list(sd.values()), list(sd.values()))
                         - target / np.outer(list(sd.values()), list(sd.values())))) < 0.06
    assert sims.info["shock_corr"] == {"ey,epi": 0.4, "epi,ei": -0.3}


def test_occbin_forecasts_with_regimes_are_the_regime_forecasts_when_nothing_binds():
    far = templates.new_keynesian(lower_bound=-100.0)
    data = _occbin_data(templates.new_keynesian(lower_bound=2.0), 60, 5, [])
    fit = far.estimate(data, ["rho"], measurement=NOISY, regimes=[Regime("1990Q1", {"phi_pi": 2.0}, name="hawk")], cov="none")
    span = ("1995Q1", "1997Q4")
    ahead = "1995Q3..1996Q2: rho=1, phi_pi=0"
    linear = fit.forecast(span, regimes=ahead, occbin=False)
    with_occbin = fit.forecast(span, regimes=ahead, occbin=True)
    assert with_occbin.provenance.method["binding"] == {"policy (max)": []}
    # the OccBin state is the OccBin filter's (the same as the regime filter's here)
    assert np.max(np.abs(with_occbin.frame.loc["1995Q1":, COLS].to_numpy() - linear.frame.loc["1995Q1":, COLS].to_numpy())) < 1e-8
    assert np.ptp(with_occbin.frame.loc["1995Q3":"1996Q2", "i"]) < 1e-10


# ---------------------------------------------------------------------------------------------
# posterior forecasts and decompositions with regimes


def _nk_data(nk, n, seed, regimes):
    rng = np.random.default_rng(seed)
    idx = _quarters(n, "1995Q1")
    shocks = pd.DataFrame(rng.normal(size=(n, 3)) * [nk.shock_sd[s] for s in nk.shocks], index=idx, columns=list(nk.shocks))
    sc = Scenario(set={s: shocks[s] for s in shocks}, unanticipated=list(nk.shocks))
    path = nk.solve().solve(None, (str(idx[0]), str(idx[-1])), sc, regimes=regimes)
    return nk.measurement.apply(Dataset.from_frame(path.frame.loc[idx, COLS]))


@pytest.fixture(scope="module")
def regime_fit():
    nk = templates.new_keynesian()
    data = _nk_data(nk, 100, 3, [Regime("2010Q1", {"phi_pi": 2.5}, name="post")])
    fit = nk.estimate(data, estimate=["phi_pi", "phi_pi@post"], regimes=[Regime("2010Q1", {"phi_pi": 1.5}, name="post")],
                      priors={"phi_pi": priors.Normal(1.5, 0.3), "phi_pi@post": priors.Normal(2.0, 0.5)})
    return nk, data, fit


def test_posterior_forecasts_with_regimes_are_each_draws_regime_forecast(regime_fit):
    nk, data, fit = regime_fit
    post = fit.mcmc(draws=60, chains=1, seed=2, workers=1)
    span, ahead = ("2020Q1", "2021Q4"), "2020Q2..2020Q3: rho=1, phi_pi=0"
    forecast = post.forecast(span, draws=4, shocks=False, seed=1, regimes=ahead)
    sims = forecast.simulation
    assert sims.values.shape[0] == 4
    for values, path in zip(post._subsample(4), sims.values):
        template = templates.new_keynesian(phi_pi=values["phi_pi"])
        regimes = [Regime("2010Q1", {"phi_pi": values["phi_pi@post"]}, name="post")]
        state = regime_filter(template, data, regimes).smoothed
        again = template.solve().solve(state, span, regimes=[*regimes, Regime("2020Q2", {"rho": 1.0, "phi_pi": 0.0}, end="2020Q3")])
        report = template.measurement.report(again, data).frame
        for j, name in enumerate(sims.names):
            expected = (again.frame if name in COLS else report)[name].loc[span[0]:span[1]].to_numpy()
            assert np.max(np.abs(path[:, j] - expected)) < 1e-10, name
    assert np.ptp(sims.values[:, 1:3, sims.names.index("i")], axis=1).max() < 1e-10  # each draw pegs the rate
    # the Forecast dialog takes regimes for a posterior
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import ForecastModel

    s = Session()
    s.objects.update({"post": post})
    cmd = ForecastModel("post", "2020Q1", "2021Q4", draws=3, shocks=False, regimes=ahead, name="fc")
    assert cmd.validate(s) == []
    s.run(cmd)
    in_code = post.forecast(span, draws=3, shocks=False, seed=0, regimes=ahead)
    assert np.array_equal(s["fc"].simulation.values, in_code.simulation.values)
    assert "regimes='2020Q2..2020Q3: rho=1, phi_pi=0'" in s.script()


def test_regime_decompositions_add_up_and_moments_are_the_regimes_own(regime_fit):
    nk, data, fit = regime_fit
    d = fit.decompose()
    assert np.max(np.abs(d.residual.to_numpy())) < 1e-10
    assert list(d.contributions["y"].columns) == ["ey", "epi", "ei", "initial conditions", "regimes"]
    again = fit.solution.decompose(fit.smoothed, regimes=fit.regimes)
    assert max(np.max(np.abs(again.contributions[v].to_numpy() - d.contributions[v].to_numpy())) for v in COLS) < 1e-10
    shocked = fit.decompose(start="presample")
    assert np.max(np.abs(shocked.residual.to_numpy())) < 1e-10 and len(shocked.deviations) == len(d.deviations) - 1
    # a regime that moves the steady state: its constant is a part of its own, and the parts still add up
    sol = nk.solve()
    sc = Scenario(set={"ey": {"2001Q2": -1.0}, "ei": {"2003Q1": 0.4}}, unanticipated=["ey", "ei"])
    target = [Regime("2002Q1", {"phi_pi": 2.0}), Regime("2004Q1", {"rho": 0.5}, end="2004Q4")]
    path = sol.solve(None, ("2001Q1", "2010Q4"), sc, regimes=target)
    by_regimes = sol.decompose(path.frame.loc["2001Q1":], regimes=target)
    assert np.max(np.abs(by_regimes.residual.to_numpy())) < 1e-10
    shocks = by_regimes.shocks
    assert abs(shocks.loc["2001Q2", "ey"] + 1.0) < 1e-10 and abs(shocks.loc["2003Q1", "ei"] - 0.4) < 1e-10
    # regimes equal to the base: the linear decomposition
    same = sol.decompose(path.frame.loc["2001Q1":], regimes=[Regime("2002Q1", {"phi_pi": 1.5})])
    plain = sol.decompose(path.frame.loc["2001Q1":])
    assert all(np.array_equal(same.contributions[v].to_numpy(), plain.contributions[v].to_numpy()) for v in COLS)
    # moments of one regime: the model at its values, on its own
    values = fit.params.as_dict()
    own = templates.new_keynesian(phi_pi=values["phi_pi@post"]).solve().moments(fit.shock_sd)
    assert np.max(np.abs(fit.moments(regime="post").covariance.to_numpy() - own.covariance.to_numpy())) < 1e-10
    assert np.max(np.abs(fit.moments().covariance.to_numpy() - templates.new_keynesian(phi_pi=values["phi_pi"]).solve().moments(fit.shock_sd).covariance.to_numpy())) < 1e-10
    with pytest.raises(ModelError, match="no regime called"):
        fit.moments(regime="nonesuch")
