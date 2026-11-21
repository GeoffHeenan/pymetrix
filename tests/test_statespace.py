"""State space: specification, the Kalman filter and smoother (native and statsmodels engines),
exact diffuse starts, estimation, unobserved components, and the dialogs."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymetrix.statespace as ss
from pymetrix import Dataset, Param, StateSpace, UnobservedComponents
from pymetrix.estimation.priors import Beta
from pymetrix.expr import ExpressionError
from pymetrix.filters import hp_trend

sm = pytest.importorskip("statsmodels.api")

DATA = Path(__file__).parents[1] / "examples" / "sample_macro.parquet"
UC = StateSpace(
    ["trend", "slope", "cycle"],
    ["trend = trend[-1] + slope[-1] + e_trend", "slope = slope[-1] + e_slope", "cycle = phi1*cycle[-1] + phi2*cycle[-2] + e_cycle"],
    ["y = trend + cycle + u"],
    {"e_trend": "sd_trend", "e_slope": "sd_slope", "e_cycle": "sd_cycle", "u": "sd_u"},
    ["phi1", "phi2", "sd_trend", "sd_slope", "sd_cycle", "sd_u"],
)
VALUES = dict(phi1=1.2, phi2=-0.4, sd_trend=0.3, sd_slope=0.05, sd_cycle=0.6, sd_u=0.2)


@pytest.fixture(scope="module")
def simulated():
    rng = np.random.default_rng(42)
    n = 240
    trend = np.cumsum(0.5 + rng.normal(0, 0.3, n))
    cyc = np.zeros(n)
    for t in range(2, n):
        cyc[t] = 1.3 * cyc[t - 1] - 0.5 * cyc[t - 2] + rng.normal(0, 0.7)
    y = trend + cyc + rng.normal(0, 0.2, n)
    y_gaps = y.copy()
    y_gaps[[40, 41, 90]] = np.nan
    frame = pd.DataFrame({"y": y, "y_gaps": y_gaps}, index=pd.period_range("1960Q1", periods=n, freq="Q", name="time"))
    return Dataset.from_frame(frame), cyc


def _sm(y, m, init):
    model = sm.tsa.statespace.MLEModel(y, k_states=len(m.states), k_posdef=len(m.shocks))
    for key, v in dict(design=m.Z, obs_intercept=m.d, obs_cov=m.H, transition=m.T, state_intercept=m.c, selection=m.R, state_cov=m.Q).items():
        model.ssm[key] = v
    init(model.ssm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.smooth([])


def test_equations_become_matrices():
    m = UC.system(VALUES)
    assert m.states == ("trend", "slope", "cycle", "cycle[-1]")  # cycle[-2] needs a lagged copy of the cycle
    assert np.allclose(m.T, [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1.2, -0.4], [0, 0, 1, 0]])
    assert np.allclose(m.R @ np.sqrt(m.Q), np.diag([0.3, 0.05, 0.6])[[0, 1, 2]].tolist() + [[0, 0, 0]])
    assert np.allclose(m.Z, [[1, 0, 1, 0]]) and np.isclose(m.H[0, 0], 0.04)
    with_lagged_measure = StateSpace(["x"], ["x = rho*x[-1] + e"], ["y = x + 0.5*x[-1]"], {"e": 1.0}, ["rho"])
    assert with_lagged_measure.system({"rho": 0.5}).states == ("x", "x[-1]")


@pytest.mark.parametrize("bad, message", [
    (dict(transition=["trend = trend[1] + e"]), "leads"),
    (dict(transition=["trend = trend + e"]), "lagged states"),
    (dict(transition=["trend = trend[-1]*trend[-1] + e"]), "not linear"),
    (dict(transition=["trend = trend[-1] + z + e"]), "'z' is not a state"),
    (dict(transition=[]), "no transition equation for trend"),
    (dict(measurement=["y = trend + e"]), "both a state shock and a measurement error"),
    (dict(shocks={"e": "sd", "unused": 1.0}), "appear in no equation"),
    (dict(shocks={"e": "nope"}), "not a declared parameter"),
])
def test_specification_errors_are_explained(bad, message):
    spec = dict(states=["trend"], transition=["trend = trend[-1] + e"], measurement=["y = trend"], shocks={"e": "sd"}, params=["sd"])
    with pytest.raises(ExpressionError, match=message):
        StateSpace(**{**spec, **bad})


def test_native_filter_and_smoother_match_statsmodels(simulated):
    data, _ = simulated
    y = data["y_gaps"].to_numpy().reshape(-1, 1)  # with missing observations
    m = UC.system(VALUES)
    a0, P0, burn = m.initial("auto", ss._scale(y))
    kf = ss.kalman_filter(y, m, a0, P0, burn)
    a_s, P_s = ss.kalman_smoother(kf, m)
    ref = _sm(y[:, 0], m, lambda ssm: (ssm.initialize_known(a0, P0), setattr(ssm, "loglikelihood_burn", burn)))
    assert np.isclose(kf.loglike, ref.llf, rtol=1e-7)  # the approximate start's large variance costs some precision
    # the approximate start's very large variance costs precision in the first periods, which fades
    assert np.allclose(kf.a_filt[burn:], ref.filtered_state.T[burn:], atol=1e-4)
    assert np.allclose(kf.a_filt[20:], ref.filtered_state.T[20:], atol=1e-5)  # how close depends on the platform's linear algebra
    assert np.allclose(a_s[20:], ref.smoothed_state.T[20:], atol=1e-7)
    later = np.s_[10:]
    assert np.allclose(P_s[later], np.moveaxis(ref.smoothed_state_cov, 2, 0)[later], atol=1e-6)


def test_exact_diffuse_likelihood_matches_statsmodels(simulated):
    data, _ = simulated
    y = data["y_gaps"].to_numpy().reshape(-1, 1)
    for values in (VALUES, dict(VALUES, phi1=0.3, phi2=0.1, sd_slope=0.3)):
        m = UC.system(values)
        ll, diffuse = ss.diffuse_loglike(y, m, "diffuse")
        ref = _sm(y[:, 0], m, lambda ssm: ssm.initialize_diffuse())
        assert np.isclose(ll, ref.llf, rtol=1e-9) and diffuse == ref.nobs_diffuse


@pytest.mark.parametrize("engine", ["native", "statsmodels"])
def test_engines_agree(simulated, engine, monkeypatch):
    data, _ = simulated
    stationary = StateSpace(["x"], ["x = rho*x[-1] + e"], ["y = mu + x + u"], {"e": "sd", "u": 0.5}, ["rho", "sd", "mu"])
    values = {"rho": 0.8, "sd": 1.0, "mu": 3.0}
    monkeypatch.setattr(ss, "ENGINE", engine)
    fit = stationary.filter(data.with_series(y=data["y"] - data["y"].mean()), values)
    monkeypatch.setattr(ss, "ENGINE", "statsmodels" if engine == "native" else "native")
    other = stationary.filter(data.with_series(y=data["y"] - data["y"].mean()), values)
    assert np.isclose(fit.stats["loglike"], other.stats["loglike"], rtol=1e-9)
    assert np.allclose(fit.smoothed["x"], other.smoothed["x"], atol=1e-8)


def test_hp_filter_is_a_smooth_trend_model():
    data = Dataset.read(str(DATA)).with_series(ly="100*log(y)")
    hp = StateSpace(["trend", "slope"], ["trend = trend[-1] + slope[-1]", "slope = slope[-1] + e_slope"], ["ly = trend + u"],
                    {"e_slope": 1.0, "u": 1600 ** 0.5})
    assert np.allclose(hp.filter(data, {}).smoothed["trend"], hp_trend(data["ly"].to_numpy(), 1600), atol=1e-6)


def test_unobserved_components_recover_parameters(simulated):
    data, cyc = simulated
    fit = UnobservedComponents("y", trend="random walk with drift", cycle=2).fit(data)
    truth = {"drift": 0.5, "phi1": 1.3, "phi2": -0.5, "sd_cycle": 0.7}
    se = dict(zip(fit.params.names, fit.params.se))
    for name, value in truth.items():
        assert abs(fit.values[name] - value) < 3 * se[name], name
    assert np.corrcoef(fit.smoothed["cycle"].to_numpy(), cyc)[0, 1] > 0.9
    assert "converged" not in fit.stats and abs(fit.innovations()["y"].std() - 1) < 0.15
    assert fit.provenance.method["backend"].startswith("statsmodels")


def test_unobserved_components_match_statsmodels_on_data():
    data = Dataset.read(str(DATA)).with_series(ly="100*log(y)")
    ours = UnobservedComponents("ly", trend="smooth trend", cycle=1).fit(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # statsmodels' default start matches ours: the trend diffuse, the autoregressive cycle stationary
        model = sm.tsa.UnobservedComponents(data["ly"].to_numpy(), level="smooth trend", autoregressive=1, irregular=True)
        ref = model.fit(disp=False, maxiter=5000, method="nm")
        ref = model.fit(ref.params, disp=False, maxiter=5000)
    p = dict(zip(model.param_names, ref.params))
    assert np.isclose(ours.values["sd_slope"] ** 2, p["sigma2.trend"], rtol=1e-3)
    assert np.isclose(ours.values["sd_cycle"] ** 2, p["sigma2.ar"], atol=1e-4) and np.isclose(ours.values["sd_irregular"] ** 2, p["sigma2.irregular"], atol=1e-4)
    theirs = dict(sd_slope=p["sigma2.trend"] ** 0.5, phi1=p["ar.L1"], sd_cycle=p["sigma2.ar"] ** 0.5, sd_irregular=p["sigma2.irregular"] ** 0.5)
    spec = UnobservedComponents("ly", trend="smooth trend", cycle=1).spec()
    assert np.isclose(spec.loglike(data, theirs) - ref.llf, -2 * 0.5 * np.log(2 * np.pi), atol=1e-3)  # statsmodels' UC omits log(2 pi) for its 2 diffuse terms


def test_priors_fixed_values_and_degenerate_models(simulated):
    data, _ = simulated
    spec = UnobservedComponents("y", trend="random walk with drift", cycle=1).spec()
    tight = StateSpace(spec.states, spec.transition, spec.measurement, spec.shocks,
                       [Param("phi1", bounds=(-0.999, 0.999), prior=Beta(0.3, 0.02)) if p.name == "phi1" else p for p in spec.params],
                       stationary=spec.stationary)
    free_fit, prior_fit = spec.fit(data), tight.fit(data)
    assert abs(prior_fit.values["phi1"] - 0.3) < abs(free_fit.values["phi1"] - 0.3)  # pulled towards the prior
    assert prior_fit.provenance.method["estimation"] == "posterior mode"
    fixed = spec.fit(data, fixed={"sd_irregular": 0.0})
    assert fixed.values["sd_irregular"] == 0.0 and "sd_irregular" not in fixed.params.names
    zero = {**free_fit.values, "sd_trend": 0.0, "sd_cycle": 0.0, "sd_irregular": 0.0}
    assert spec.loglike(data, zero) == -np.inf  # a deterministic model has no likelihood


def test_forecasts_responses_and_simulation(simulated):
    data, _ = simulated
    fit = UnobservedComponents("y", trend="random walk with drift", cycle=2).fit(data, span=(None, "2015Q4"))
    fc = fit.forecast(data, ("2016Q1", "2017Q4"))
    assert len(fc) == 8 and set(fc.names) >= {"y", "y_se", "y_lower", "y_upper", "trend", "cycle"}
    assert np.all(np.diff(fc["y_se"].to_numpy()) > 0)  # uncertainty grows with the horizon
    assert np.isclose(fc["trend"].iloc[1] - fc["trend"].iloc[0], fit.values["drift"])  # the trend rises by the drift
    irf = fit.irf(10)
    assert np.isclose(irf[("e_cycle", "cycle")].iloc[0], fit.values["sd_cycle"]) and abs(irf[("e_cycle", "cycle")].iloc[-1]) < 0.5
    sim = fit.simulate(50, seed=3)
    assert list(sim.columns) == ["trend", "cycle", "y"] and np.allclose(sim, fit.simulate(50, seed=3))


def test_dialogs_record_replay_and_add_results(simulated, tmp_path):
    from pymetrix.commands import AddToDataset, FitStateSpace, FitUnobservedComponents, Forecast, Session, load_project, save_project

    data, _ = simulated
    s = Session()
    s.objects["data"] = data
    s.run(FitUnobservedComponents("y", trend="random walk with drift", cycle=2, end="2015Q4"))
    s.run(FitStateSpace(states="level, gap", transition="level = level[-1] + drift + e_level\ngap = rho*gap[-1] + e_gap",
                        measurement="y = level + gap", shocks="e_level = sd_level, e_gap = sd_gap",
                        params="drift, rho, sd_level, sd_gap", stationary="gap", name="ssm"))
    s.run(AddToDataset("uc", "smoothed:cycle", series_name="cycle_hat"))
    s.run(Forecast("uc", "2016Q1", "2016Q4", name="fc"))
    script = s.script()
    assert "uc = UnobservedComponents(series='y', trend='random walk with drift').fit(data, span=(None, '2015Q4'))" in script
    assert "ssm = StateSpace(" in script and "stationary=['gap']" in script
    assert "data = data.with_series(cycle_hat=uc.smoothed['cycle'])" in script
    ns = {"data": data}
    exec(script, ns)
    assert np.allclose(ns["ssm"].smoothed["gap"], s["ssm"].smoothed["gap"])
    reopened = load_project(save_project(s, tmp_path / "ssm.pymx"))
    assert np.allclose(reopened["uc"].smoothed["trend"], s["uc"].smoothed["trend"])
    assert "cycle_hat" in reopened["data"] and len(reopened["fc"]) == 4
