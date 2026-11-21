"""Surprise regimes with the nonlinear solutions, regimes that change a constraint, and identification with regimes
(pymetrix.models.regimes, perfect_foresight, occbin, bayes.regime_moments).

Checked independently:
- perfect foresight with a surprise regime (expected=False) is, before the surprise starts, the path without it, and
  from then on the path with the regime known, solved by stacked Newton from the state reached when it starts (1e-8);
  with the regime expected instead, the path before it differs (the surprise matters); with several surprises (one
  temporary, believed permanent until it ends) and an expected regime among them, perfect foresight on a model whose
  equations are linear is the first-order solution with the same regimes, worked out by its own time-varying rules
  (1e-8), also together with an unanticipated shock;
- OccBin with a surprise regime is perfect foresight on the model with its max: with the shocks known in advance,
  against the path without the regime spliced with the path from the state at the surprise, each solved by stacked
  Newton with the regime's parameter as a series in the data (1e-8); with shocks that are surprises too, against
  perfect foresight with the same surprise regime and unanticipated shocks (1e-8); the OccBin filter with a surprise
  regime, where nothing binds, is the (first-order) regime filter with it (likelihood and smoothed path, 1e-8), and the
  regime made expected gives another likelihood;
- OccBin regimes that switch a constraint off for a while, rewrite its floor, or bring one into a model without it are
  perfect foresight on a model with the floor as a series in the data (a floor far below for "off", none in the
  model's own equation), solved by stacked Newton (1e-8), the constraint binding where it would otherwise;
- identification with regimes equal to the base is the identification without regimes, bit for bit; a regime's
  parameter is not identified when the regime starts after the sample, and is when it starts inside it; the moments
  through a permanent regime settle, by the sample's end, at that regime's own stationary moments (1e-8); a fit
  estimated with regimes is checked as the model is at the fit's estimates; the Solve dialog takes a surprise regime
  and the Identification dialog a model estimated with regimes, each giving what its recorded code gives."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, Model, Regime, occbin, parse_regimes, templates
from pymetrix.metadata import Role
from pymetrix.models import ModelError, PerfectForesight, Scenario
from pymetrix.models.bayes import moments, regime_moments

COLS = ["y", "pi", "i"]
SH = ("ey", "epi", "ei")
PF = PerfectForesight(terminal="steady_state")


def _nk():
    eqs = [Equation("y ~ y[+1] - (1/sigma)*(i - pi[+1]) + ey", params=["sigma"], name="IS"),
           Equation("pi ~ beta*pi[+1] + kappa*y + epi", params=["beta", "kappa"], name="Phillips"),
           Equation("i ~ rho*i[-1] + (1 - rho)*(pistar + phi_pi*(pi - pistar) + phi_y*y) + ei",
                    params=["rho", "phi_pi", "phi_y", "pistar"], name="policy")]
    return Model(eqs, params=dict(sigma=1.0, beta=0.99, kappa=0.1, rho=0.8, phi_pi=1.5, phi_y=0.5, pistar=0.0),
                 roles={s: Role.SHOCK for s in SH})


def _nk_data(last="2079Q4"):
    idx = pd.period_range("2024Q4", last, freq="Q", name="time")
    f = pd.DataFrame(0.0, index=idx, columns=[*COLS, *SH])
    f.loc["2024Q4", COLS] = [0.3, 0.2, 0.4]
    f.loc["2025Q1":, COLS] = np.nan
    f.loc["2025Q2", "ey"], f.loc["2027Q1", "ei"] = -1.0, 0.3
    return Dataset.from_frame(f)


def _cols(sol, first, last):
    return sol.frame.loc[first:last, COLS].to_numpy(float)


# ---------------------------------------------------------------------------------------------
# surprise regimes with perfect foresight


def test_a_surprise_regime_with_perfect_foresight_is_the_path_without_it_then_the_path_with_it():
    m, data, span = _nk(), _nk_data(), ("2025Q1", "2079Q4")
    surprise = m.solve(data, span, solver=PF, regimes=[Regime("2026Q1", {"phi_pi": 2.5}, expected=False, name="hawk")])
    without = m.solve(data, span, solver=PF)
    assert np.max(np.abs(_cols(surprise, "2025Q1", "2025Q4") - _cols(without, "2025Q1", "2025Q4"))) < 1e-8
    frame = data.frame.copy()
    frame.loc["2025Q1":"2025Q4", COLS] = without.frame.loc["2025Q1":"2025Q4", COLS]  # where the surprise finds the economy
    after = m.solve(Dataset.from_frame(frame), ("2026Q1", "2079Q4"), solver=PF, regimes=[Regime("2026Q1", {"phi_pi": 2.5})])
    assert np.max(np.abs(_cols(surprise, "2026Q1", "2060Q4") - _cols(after, "2026Q1", "2060Q4"))) < 1e-8
    expected = m.solve(data, span, solver=PF, regimes=[Regime("2026Q1", {"phi_pi": 2.5}, name="hawk")])
    assert np.max(np.abs(_cols(expected, "2025Q1", "2025Q4") - _cols(without, "2025Q1", "2025Q4"))) > 1e-3
    method = surprise.provenance.method
    assert method["surprise regimes at"] == ["2026Q1"]
    assert [s["regime"] for s in method["regimes"]] == ["base", "hawk"]


def test_perfect_foresight_surprises_are_the_first_order_surprises_on_a_linear_model():
    m, data, span = _nk(), _nk_data(), ("2025Q1", "2079Q4")
    regimes = [Regime("2025Q3", {"phi_pi": 2.0}, name="hawk"),
               Regime("2026Q1", {"rho": 0.5, "phi_pi": 3.0}, end="2026Q4", name="tight", expected=False),
               Regime("2028Q1", {"pistar": 0.5}, name="target", expected=False),
               Regime("2031Q1", equations={"policy": "i = 0.5*i[-1] + 0.5*(2*pi + 0.2*y)"}, name="alternative")]
    pf = m.solve(data, span, solver=PF, regimes=regimes)
    first_order = m.linearize(shocks=list(SH)).solve().solve(data, span, regimes=regimes)
    assert np.max(np.abs(_cols(pf, "2025Q1", "2050Q4") - _cols(first_order, "2025Q1", "2050Q4"))) < 1e-8
    assert pf.provenance.method["surprise regimes at"] == ["2026Q1", "2027Q1", "2028Q1"]  # the temporary one's end too
    # with an unanticipated shock as well: solved again at each kind of news
    sc = Scenario(set={"epi": {"2027Q3": 0.3, "2029Q2": -0.4}}, unanticipated=["epi"])
    pf_sc = m.solve(data, span, sc, solver=PF, regimes=regimes)
    fo_sc = m.linearize(shocks=list(SH)).solve().solve(data, span, sc, regimes=regimes)
    assert np.max(np.abs(_cols(pf_sc, "2025Q1", "2050Q4") - _cols(fo_sc, "2025Q1", "2050Q4"))) < 1e-8
    assert np.max(np.abs(_cols(pf_sc, "2029Q2", "2030Q4") - _cols(pf, "2029Q2", "2030Q4"))) > 1e-3


def test_regimes_as_text_take_surprises():
    hawk, = parse_regimes("2026Q1 surprise: phi_pi=2")
    assert not hawk.expected and hawk.params == {"phi_pi": 2.0}
    peg, = parse_regimes("2025Q1..2026Q4 surprise: rho=1")
    assert not peg.expected and peg.end == "2026Q4"
    assert parse_regimes("2026Q1: phi_pi=2")[0].expected


def test_the_solve_dialog_takes_a_surprise_regime_with_perfect_foresight():
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import DefineEquation, SolveModel

    s = Session()
    s.run(DefineEquation("y ~ y[+1] - s*(i - pi[+1]) + ey", "s", name="eq_is"))
    s.run(DefineEquation("pi ~ b*pi[+1] + k*y + epi", "b, k", name="eq_pc"))
    s.run(DefineEquation("i ~ r*i[-1] + (1-r)*(phi*pi + 0.5*y) + ei", "r, phi", name="eq_tr"))
    s.objects["data"] = _nk_data("2044Q4")
    cmd = SolveModel("eq_is, eq_pc, eq_tr", "2025Q1", "2044Q4", solver="perfect_foresight", solver_settings="terminal='steady_state'",
                     params="s=1, b=0.99, k=0.1, r=0.8, phi=1.5", regimes="2026Q1 surprise: phi=2.5")
    assert cmd.validate(s) == []
    s.run(cmd)
    assert "regimes='2026Q1 surprise: phi=2.5'" in s.script()
    model = Model([s["eq_is"], s["eq_pc"], s["eq_tr"]], params={"s": 1.0, "b": 0.99, "k": 0.1, "r": 0.8, "phi": 1.5})
    in_code = model.solve(s["data"], ("2025Q1", "2044Q4"), solver=PF, regimes="2026Q1 surprise: phi=2.5")
    assert np.array_equal(s["solution"].frame[COLS].to_numpy(), in_code.frame[COLS].to_numpy())
    assert in_code.provenance.method["surprise regimes at"] == ["2026Q1"]


# ---------------------------------------------------------------------------------------------
# OccBin: surprise regimes, and regimes that change a constraint

ZSH = ["ey", "epi", "ez"]


def _zlb(policy="i = max(ishadow, 0)", phi="phi"):
    """A New Keynesian model with a natural rate and a lower bound; ``phi="phi_t"``: the response a series in the data."""
    shadow = Equation(f"ishadow = rn + {phi}*pi + 0.5*y") if phi != "phi" else Equation("ishadow ~ rn + phi*pi + 0.5*y", params=["phi"])
    eqs = [Equation("y ~ y[+1] - sig*(i - pi[+1] - rn) + ey", params=["sig"]),
           Equation("pi ~ bet*pi[+1] + kap*y + epi", params=["bet", "kap"]), shadow,
           Equation(policy, name="lower bound"), Equation("rn ~ (1-rho)*rbar + rho*rn[-1] + ez", params=["rho", "rbar"])]
    params = dict(sig=1.0, bet=0.99, kap=0.1, rho=0.8, rbar=1.0, **({"phi": 1.5} if phi == "phi" else {}))
    return Model(eqs, params=params, roles={s: Role.SHOCK for s in ZSH})


def _zlb_frame(shocks, n, series=None):
    ss = _zlb().steady_state()
    idx = pd.period_range("2000Q1", periods=n + 121, freq="Q", name="time")
    frame = pd.DataFrame({v: ss[v] for v in _zlb().endogenous}, index=idx)
    for s in ZSH:
        frame[s] = 0.0
    for name, path in shocks.items():
        for t, value in path.items():
            frame.loc[idx[t], name] = value
    for name, (base, changes) in (series or {}).items():
        frame[name] = base
        for (a, b), value in changes:
            frame.loc[idx[a]:(idx[b] if b is not None else None), name] = value
    return frame, idx


def _pf(model, frame, first, idx, scenario=None, regimes=None):
    return model.solve(Dataset.from_frame(frame), (idx[first], idx[-1]), scenario, solver=PF, regimes=regimes)


def test_occbin_with_a_surprise_regime_is_perfect_foresight_solved_again_when_it_starts():
    m, n = _zlb(), 40
    shocks = {"ez": {1: -2.0}, "ey": {3: -0.6, 10: -0.8}}
    path = occbin(m, shocks, periods=n, anticipated=True, shock_names=ZSH, horizon=200,
                  regimes=[Regime(3, {"phi": 2.5}, expected=False, name="hawk")])
    # independently: stacked Newton without the regime, kept for periods 1-2; then from the state reached, the
    # response a series stepping to 2.5, the same shocks still known
    frame, idx = _zlb_frame(shocks, n, {"phi_t": (1.5, [])})
    before = _pf(_zlb(phi="phi_t"), frame, 1, idx)
    restart = frame.copy()
    restart.loc[idx[1]:idx[2], before.frame.columns.intersection(m.endogenous)] = before.frame.loc[idx[1]:idx[2], m.endogenous]
    restart["phi_t"] = 2.5
    after = _pf(_zlb(phi="phi_t"), restart, 3, idx)
    spliced = pd.concat([before.frame.loc[idx[1]:idx[2]], after.frame.loc[idx[3]:idx[n]]])
    for v in ("y", "pi", "i", "rn"):
        assert np.max(np.abs(path.frame[v].to_numpy() - spliced[v].to_numpy())) < 1e-8, v
    assert path.binding.iloc[:, 0].iloc[1] and path.binding.iloc[:, 0].iloc[2]  # binding across the surprise
    foreseen = occbin(m, shocks, periods=n, anticipated=True, shock_names=ZSH, horizon=200, regimes=[Regime(3, {"phi": 2.5})])
    assert np.max(np.abs(foreseen.frame["y"].iloc[:2].to_numpy() - path.frame["y"].iloc[:2].to_numpy())) > 1e-4
    # surprise shocks and the surprise regime: perfect foresight with the max, the same surprise regime and the shocks
    # unanticipated (solved again at each kind of news)
    repeated = occbin(m, shocks, periods=n, shock_names=ZSH, horizon=200, regimes=[Regime(3, {"phi": 2.5}, expected=False)])
    plain, idx = _zlb_frame({}, n)
    sc = Scenario(set={"ez": {str(idx[1]): -2.0}, "ey": {str(idx[3]): -0.6, str(idx[10]): -0.8}}, unanticipated=["ez", "ey"])
    pf = _pf(m, plain, 1, idx, sc, [Regime(str(idx[3]), {"phi": 2.5}, expected=False)])
    for v in ("y", "pi", "i"):
        assert np.max(np.abs(repeated.frame[v].to_numpy() - pf.frame.loc[idx[1]:idx[n], v].to_numpy())) < 1e-8, v


def test_occbin_regimes_that_switch_rewrite_or_bring_in_a_constraint_are_the_floor_as_data():
    """The floor as a series ``fl`` in the data: -1000 while the constraint is off (it never binds), -0.5 rewritten."""
    n = 40
    shocks = {"ez": {1: -2.0}, "ey": {9: -3.0}}
    regimes = [Regime(3, equations={"lower bound": "i = ishadow"}, end=5, name="off"),
               Regime(9, equations={"lower bound": "i = max(ishadow, -0.5)"}, name="lower")]
    path = occbin(_zlb(), shocks, periods=n, anticipated=True, shock_names=ZSH, horizon=200, regimes=regimes)
    frame, idx = _zlb_frame(shocks, n, {"fl": (0.0, [((3, 5), -1000.0), ((9, None), -0.5)])})
    pf = _pf(_zlb("i = max(ishadow, fl)"), frame, 1, idx)
    for v in ("y", "pi", "i", "rn"):
        assert np.max(np.abs(path.frame[v].to_numpy() - pf.frame.loc[idx[1]:idx[n], v].to_numpy())) < 1e-8, v
    binds = path.binding.iloc[:, 0]
    assert binds.iloc[0] and not binds.iloc[2:5].any() and (path.frame["i"].iloc[2:5] < -1e-3).all()  # off: below the floor
    assert (path.frame["i"].iloc[8:] >= -0.5 - 1e-10).all() and binds.iloc[8:].any()  # the lower floor, binding
    # the same regimes with a perfect-foresight model that has its max: the regimes' own equations switched in
    by_regimes = _pf(_zlb(), frame.drop(columns=["fl"]), 1, idx,
                     regimes=[Regime(str(idx[3]), equations={"lower bound": "i = ishadow"}, end=str(idx[5])),
                              Regime(str(idx[9]), equations={"lower bound": "i = max(ishadow, -0.5)"})])
    assert np.max(np.abs(by_regimes.frame.loc[idx[1]:idx[n], "i"].to_numpy() - path.frame["i"].to_numpy())) < 1e-8
    # a model without a constraint, a regime bringing one in from period 3
    brought = occbin(_zlb("i = ishadow"), shocks, periods=n, anticipated=True, shock_names=ZSH, horizon=200,
                     regimes=[Regime(3, equations={"lower bound": "i = max(ishadow, 0)"}, name="floor")])
    frame, idx = _zlb_frame(shocks, n, {"fl": (-1000.0, [((3, None), 0.0)])})
    pf = _pf(_zlb("i = max(ishadow, fl)"), frame, 1, idx)
    for v in ("y", "pi", "i"):
        assert np.max(np.abs(brought.frame[v].to_numpy() - pf.frame.loc[idx[1]:idx[n], v].to_numpy())) < 1e-8, v
    assert brought.frame["i"].iloc[0] < -1e-3 and brought.binding.iloc[2:, 0].any()
    with pytest.raises(ModelError, match="neither the model nor its regimes"):
        occbin(_zlb("i = ishadow"), shocks, periods=n, shock_names=ZSH, regimes=[Regime(3, {"phi": 2.0})])


def test_the_occbin_filter_with_a_surprise_regime_is_the_regime_filter_when_nothing_binds():
    from pymetrix import occbin_filter, regime_filter
    from pymetrix.models import MeasurementMap

    noisy = MeasurementMap.direct(COLS, {"y": 0.02, "pi": 0.02, "i": 0.02})
    far = templates.new_keynesian(lower_bound=-100.0)
    rng = np.random.default_rng(5)
    idx = pd.period_range("1980Q1", periods=60, freq="Q", name="time")
    data = Dataset.from_frame(pd.DataFrame(rng.normal(0, 0.3, (60, 3)), index=idx, columns=COLS))
    regimes = [Regime("1985Q1", {"phi_pi": 2.5}, expected=False, name="hawk"), Regime("1990Q1", {"phi_y": 0.1}, name="soft")]
    result = occbin_filter(far, data, noisy, regimes=regimes)
    linear = regime_filter(far.model, data, regimes, measurement=noisy, shock_sd=far.shock_sd, shocks=far.shocks)
    assert not result.binding.to_numpy().any()
    assert result.loglike == pytest.approx(linear.loglike, rel=1e-9)
    assert np.max(np.abs(result.smoothed.frame[COLS].to_numpy() - linear.smoothed.frame[COLS].to_numpy())) < 1e-8
    expected = regime_filter(far.model, data, [Regime("1985Q1", {"phi_pi": 2.5}), regimes[1]], measurement=noisy,
                             shock_sd=far.shock_sd, shocks=far.shocks)
    assert abs(expected.loglike - linear.loglike) > 1e-6


# ---------------------------------------------------------------------------------------------
# identification with regimes


def _data(nk, n=120, seed=4, regimes=()):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1990Q1", periods=n, freq="Q", name="time")
    shocks = pd.DataFrame(rng.normal(size=(n, 3)) * [nk.shock_sd[s] for s in nk.shocks], index=idx, columns=list(nk.shocks))
    sc = Scenario(set={s: shocks[s] for s in shocks}, unanticipated=list(nk.shocks))
    path = nk.solve().solve(None, (str(idx[0]), str(idx[-1])), sc, regimes=list(regimes) or None)
    return nk.measurement.apply(Dataset.from_frame(path.frame.loc[idx, COLS]))


@pytest.fixture(scope="module")
def nk():
    return templates.new_keynesian()


def test_identification_with_regimes_equal_to_the_base_is_the_identification_without(nk):
    data = _data(nk)
    plain = nk.estimate(data, estimate=["kappa", "phi_pi"], gradient="numerical")  # as regimes have (no analytic gradient)
    same = nk.estimate(data, estimate=["kappa", "phi_pi"], regimes=[Regime("2000Q1", {"phi_y": nk.model.parameters["phi_y"].value})])
    a, b = plain.identification(), same.identification()
    assert np.array_equal(a.jacobian, b.jacobian) and a.rank == b.rank
    assert np.array_equal(a.sensitivity, b.sensitivity) and np.array_equal(a.collinearity, b.collinearity)
    ea, eb = plain.identification(elasticity=True), same.identification(elasticity=True)
    assert np.array_equal(ea.jacobian, eb.jacobian)
    m, mr = nk.identification(["kappa", "phi_pi"]), nk.identification(["kappa", "phi_pi"], regimes="2000Q1: phi_y=0.5",
                                                                        span=("1990Q1", "2019Q4"))
    assert np.array_equal(m.jacobian, mr.jacobian)


def test_a_regime_parameter_is_identified_only_when_its_regime_is_in_the_sample(nk):
    span = ("1990Q1", "2019Q4")
    never = nk.identification(["kappa", "phi_pi", "phi_pi@late"], regimes=[Regime("2030Q1", {"phi_pi": 2.5}, name="late")], span=span)
    assert "phi_pi@late" in never.problems and not never.identified
    assert never.sensitivity[2] < 1e-8 * np.max(never.sensitivity) and "kappa" not in never.problems
    inside = nk.identification(["kappa", "phi_pi", "phi_pi@late"], regimes=[Regime("2005Q1", {"phi_pi": 2.5}, name="late")], span=span)
    assert inside.identified and not inside.problems and inside.sensitivity[2] > 1e-3
    assert nk.identification(["kappa", "phi_pi", "phi_pi@late"], regimes=[Regime("2005Q1", {"phi_pi": 2.5}, name="late")], span=span,
                             elasticity=True).identified
    with pytest.raises(Exception, match="span"):
        nk.model.identification(nk.measurement, ["kappa"], nk.shocks, nk.shock_sd, regimes="2005Q1: phi_pi=2")


def test_moments_through_a_permanent_regime_settle_at_the_regimes_own(nk):
    """The last period's block of the stacked moments, 150 periods after the regime starts, against the moments of
    the model solved at the regime's values on its own (its stationary distribution)."""
    sol = nk.solve()
    system = nk.system()
    periods = pd.period_range("1990Q1", periods=160, freq="Q")
    stacked = regime_moments(sol, system, [Regime("1992Q3", {"phi_pi": 2.5, "rho": 0.6})], periods, lags=2)
    own = nk.model.with_parameters({"phi_pi": 2.5, "rho": 0.6}).linearize(nk.steady_state(), shocks=nk.shocks).solve()
    target = moments(own.system(nk.measurement, nk.shock_sd), 2)
    assert np.max(np.abs(stacked[-len(target):] * np.sqrt(len(periods)) - target)) < 1e-8 * max(1.0, np.max(np.abs(target)))
    first = moments(system, 2)
    assert np.max(np.abs(stacked[:3] * np.sqrt(len(periods)) - first[:3])) < 1e-10  # the means, before anything changes
    assert np.max(np.abs(target - first)) > 1e-2


def test_a_fit_with_regimes_is_checked_as_the_model_and_by_the_dialog(nk):
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import CheckIdentification

    regimes = [Regime("2005Q1", {"phi_pi": 2.0}, name="post")]
    data = _data(nk, regimes=regimes)
    fit = nk.estimate(data, estimate=["kappa", "phi_pi", "phi_pi@post"], regimes=regimes)
    ident = fit.identification()
    assert ident.identified
    at = fit.params.as_dict()
    model = nk.identification(["kappa", "phi_pi", "phi_pi@post"], at=at, regimes=regimes, span=("1990Q1", "2019Q4"))
    assert np.allclose(ident.jacobian, model.jacobian, rtol=1e-5, atol=1e-8)
    s = Session()
    s.objects["fit"] = fit
    cmd = CheckIdentification("fit", lags=4)
    assert cmd.validate(s) == []
    s.run(cmd)
    assert np.array_equal(s["identification"].jacobian, fit.identification(lags=4).jacobian)
