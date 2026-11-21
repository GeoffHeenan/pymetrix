"""Correlated shocks: ``corr`` in Dynare files (shocks block and estimated_params), and ``shock_corr`` for any model.

Checked against: the log likelihood of statsmodels' Kalman filter on the same state-space system (whose shock
covariance has the correlation off its diagonal), the covariance ``D C D`` built by hand; draws' sample covariance
against the one given (within sampling error); impulse responses and moments against ``Q chol(Σ)`` and the Lyapunov
equation with the full covariance; a correlation estimated from simulated data against the one that made them
(within three standard errors); without correlations, results identical to the arithmetic with independent shocks.
The Dynare writer writes ``corr`` and reads back the same; refusals for unknown shocks, values outside (-1, 1) and
correlations that are not positive definite."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import linalg

from pymetrix import Dataset, Equation, Model, ModelSource, read_mod, templates
from pymetrix.estimation.priors import Normal
from pymetrix.metadata import Param, Role
from pymetrix.models import ModelError
from pymetrix.models.writers import export_model

sys.path.insert(0, str(Path(__file__).parent))
from test_dynare import NK  # noqa: E402

NK_CORR = NK.replace("var ei; stderr 0.2; end;", "var ei; stderr 0.2; corr ey, ei = 0.4; end;")

TWO = """
var x1 x2;
varexo e1 e2;
parameters r1 r2;
r1 = 0.5; r2 = 0.7;
model;
x1 = r1*x1(-1) + e1;
x2 = r2*x2(-1) + e2;
end;
shocks; var e1; stderr 1; var e2; stderr 0.5; corr e1, e2 = 0.5; end;
varobs x1 x2;
"""


def _quarters(n: int) -> pd.PeriodIndex:
    return pd.period_range("1950Q1", periods=n, freq="Q", name="time")


def _simulated(dyn, n: int, seed: int) -> Dataset:
    frame = dyn.solve().simulate(n, dyn.shock_sd, seed=seed, shock_corr=dyn.shock_corr)
    frame.index = _quarters(n)
    return Dataset.from_frame(frame)


def test_the_shocks_block_reads_corr_and_covariances():
    dyn = read_mod(NK_CORR)
    assert dyn.shock_corr == {("ey", "ei"): 0.4} and dyn.shock_sd["ey"] == 0.5
    assert read_mod(NK_CORR.replace("corr ey, ei", "corr ei, ey")).shock_corr == {("ey", "ei"): 0.4}  # the shocks' order
    cov = read_mod(NK.replace("var ei; stderr 0.2; end;", "var ei; stderr 0.2; var epi, ei = 0.3*0.3*0.2; end;"))
    assert cov.shock_corr == {("epi", "ei"): pytest.approx(0.3)}  # covariance / (sd_epi sd_ei), sd_epi from var epi = 0.09
    before = read_mod(NK.replace("shocks; var ey;", "shocks; corr ey, epi = -0.2; var ey;"))  # anywhere in the block
    assert before.shock_corr == {("ey", "epi"): -0.2}
    assert read_mod(NK).shock_corr == {}


def test_the_state_space_covariance_and_statsmodels_likelihood():
    mlemodel = pytest.importorskip("statsmodels.tsa.statespace.mlemodel")
    dyn = read_mod(NK_CORR)
    sol = dyn.solve()
    system = sol.system(["y", "pi", "i"], dyn.shock_sd, shock_corr=dyn.shock_corr)
    sd = np.array([0.5, 0.3, 0.2])
    C = np.eye(3)
    C[0, 2] = C[2, 0] = 0.4
    np.testing.assert_allclose(system.Q, np.diag(sd) @ C @ np.diag(sd), atol=1e-15)  # D C D, by hand
    assert system.Q[0, 2] == pytest.approx(0.4 * 0.5 * 0.2) and system.shock_corr == {("ey", "ei"): 0.4}
    data = _simulated(dyn, 200, seed=3)
    y, index = system.observations(data)
    ours = system.loglike_observed(y, index, start="stationary")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = mlemodel.MLEModel(y - system.d, k_states=len(system.states), k_posdef=len(system.shocks))
        ref.ssm["design"], ref.ssm["obs_cov"] = system.Z, system.H
        ref.ssm["transition"], ref.ssm["selection"], ref.ssm["state_cov"] = system.T, system.R, system.Q
        ref.ssm.initialize_stationary()
        theirs = ref.ssm.loglike()
    assert ours == pytest.approx(theirs, rel=1e-9, abs=1e-8)
    plain = sol.system(["y", "pi", "i"], dyn.shock_sd)
    assert plain.loglike_observed(y, index, start="stationary") != pytest.approx(ours, rel=1e-6)  # the correlation matters
    # zero correlations are no correlations: the same matrices, the same likelihood, to the last bit
    zero = sol.system(["y", "pi", "i"], dyn.shock_sd, shock_corr={("ey", "ei"): 0.0})
    assert np.array_equal(zero.Q, plain.Q) and zero.loglike_observed(y, index) == plain.loglike_observed(y, index)
    assert read_mod(NK_CORR + "varobs y pi i;").log_posterior(data=data) == pytest.approx(system.loglike_observed(y, index), rel=1e-12)  # the file's corr used


def test_draws_have_the_covariance_given():
    static = Model([Equation("a = e1"), Equation("b = e2"), Equation("c = e3")], roles={s: Role.SHOCK for s in ("e1", "e2", "e3")})
    sol = static.linearize(shocks=["e1", "e2", "e3"]).solve()
    sd = {"e1": 1.0, "e2": 2.0, "e3": 0.5}
    corr = {("e1", "e2"): 0.6, ("e3", "e1"): -0.3}
    sigma = np.array([[1.0, 0.6 * 2, -0.3 * 0.5], [0.6 * 2, 4.0, 0.0], [-0.3 * 0.5, 0.0, 0.25]])
    data = Dataset.from_frame(pd.DataFrame({"a": [0.0] * 4, "b": [0.0] * 4, "c": [0.0] * 4}, index=_quarters(4)))
    path = sol.solve(data, ("1950Q3", "1950Q4"), draws=20000, shock_sd=sd, seed=5, shock_corr=corr)
    draws = path.simulation.values.reshape(-1, 3)
    se = np.sqrt((sigma**2 + np.outer(np.diag(sigma), np.diag(sigma))) / len(draws))  # of each sample covariance
    assert np.all(np.abs(np.cov(draws.T) - sigma) < 4 * se)
    assert path.simulation.info["shock_corr"] == {"e1,e2": 0.6, "e1,e3": -0.3}
    frame = sol.simulate(40000, sd, seed=2, levels=False, shock_corr=corr)
    assert np.all(np.abs(np.cov(frame.to_numpy().T) - sigma) < 4 * np.sqrt((sigma**2 + np.outer(np.diag(sigma), np.diag(sigma))) / 40000))
    model_draws = static.solve(data, ("1950Q3", "1950Q4"), solver=__import__("pymetrix").models.FirstOrder(), draws=5000,
                               shock_sd=sd, seed=5, shock_corr=corr)
    assert model_draws.simulation.info["shock_corr"] == {"e1,e2": 0.6, "e1,e3": -0.3}
    with pytest.raises(Exception, match="first-order solution"):
        static.solve(data, ("1950Q3", "1950Q4"), shock_corr=corr)


def test_responses_and_moments_orthogonalised_as_dynare():
    dyn = read_mod(NK_CORR)
    sol = dyn.solve()
    sd = np.array([0.5, 0.3, 0.2])
    sigma = np.diag(sd**2)
    sigma[0, 2] = sigma[2, 0] = 0.4 * 0.5 * 0.2
    impact = sol.Q @ np.linalg.cholesky(sigma)
    irf = sol.irf(5, dyn.shock_sd, dyn.shock_corr)
    k = len(sol.endogenous)
    np.testing.assert_allclose(irf.values[0], impact[:k], atol=1e-14)
    np.testing.assert_allclose(irf.values[2], (sol.P @ sol.P @ impact)[:k], atol=1e-14)
    assert "Cholesky" in irf.identification
    unit = sol.irf(5)  # without sizes or correlations: unit shocks, as always
    np.testing.assert_array_equal(unit.values[0], sol.Q[:k])
    moments = sol.moments(dyn.shock_sd, shock_corr=dyn.shock_corr)
    S = linalg.solve_discrete_lyapunov(sol.P, sol.Q @ sigma @ sol.Q.T)
    np.testing.assert_allclose(moments.covariance.to_numpy(), S[:k, :k], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(moments.shares.sum(axis=1), 1.0, atol=1e-10)  # orthogonalised parts add up
    assert "Cholesky" in str(moments.table()) and "ey-ei 0.4" in str(moments.summary())
    fevd = sol.fevd(8, dyn.shock_sd, dyn.shock_corr)
    np.testing.assert_allclose(fevd.values.sum(axis=2), 1.0, atol=1e-10)
    independent = sol.moments(dyn.shock_sd)
    assert not np.allclose(independent.covariance.to_numpy(), moments.covariance.to_numpy())
    second = dyn.solve(order=2)
    np.testing.assert_allclose(second.sigma, sigma, atol=1e-15)  # the risk terms take the full covariance


def test_estimating_a_correlation_recovers_it():
    dyn = read_mod(TWO)
    data = _simulated(dyn, 600, seed=7)
    text = TWO + "estimated_params;\nr1, 0.4, 0, 0.99;\nstderr e1, 0.8, 0.01, 5;\nstderr e2, 0.8, 0.01, 5;\ncorr e1, e2, 0, -1, 1;\nend;\n"
    fit = read_mod(text).estimate(data, start="stationary")
    est = dict(zip(fit.params.names, fit.params.values))
    se = dict(zip(fit.params.names, fit.params.se))
    assert fit.params.names == ("r1", "sd_e1", "sd_e2", "corr_e1_e2")
    for name, truth in {"corr_e1_e2": 0.5, "r1": 0.5, "sd_e1": 1.0, "sd_e2": 0.5}.items():
        assert abs(est[name] - truth) < 3 * se[name], name
    assert fit.shock_corr == {("e1", "e2"): pytest.approx(est["corr_e1_e2"])}
    assert fit.system.Q[0, 1] == pytest.approx(est["corr_e1_e2"] * est["sd_e1"] * est["sd_e2"])
    assert fit.bounds["corr_e1_e2"] == (-1.0, 1.0)
    # the same through Model.estimate: shock_corr given is estimated by default, with a prior from a Param spec
    model = Model(dyn.model.equations, params=dyn.model.parameters, specs=[Param("corr_e1_e2", prior=Normal(0.0, 0.5))])
    generic = model.estimate(data, ["x1", "x2"], shocks=["e1", "e2"], shock_sd=dyn.shock_sd, shock_corr={("e1", "e2"): 0.2},
                             start="stationary")
    assert "corr_e1_e2" in generic.params.names and generic.method == "posterior mode"
    assert generic.priors["corr_e1_e2"] == Normal(0.0, 0.5)
    g = dict(zip(generic.params.names, generic.params.values))
    assert abs(g["corr_e1_e2"] - 0.5) < 3 * dict(zip(generic.params.names, generic.params.se))["corr_e1_e2"]
    assert generic.moments().shock_corr == generic.shock_corr  # the fit's moments with its correlations


def test_without_correlations_estimates_are_unchanged():
    dyn = read_mod(TWO.replace("corr e1, e2 = 0.5;", ""))
    data = _simulated(dyn, 200, seed=4)
    a = dyn.model.estimate(data, ["x1", "x2"], shocks=["e1", "e2"], shock_sd=dyn.shock_sd)
    b = dyn.model.estimate(data, ["x1", "x2"], shocks=["e1", "e2"], shock_sd=dyn.shock_sd, shock_corr={("e1", "e2"): 0.0},
                           estimate=list(a.params.names))
    assert np.array_equal(a.params.values, b.params.values) and a.stats["loglike"] == b.stats["loglike"]
    assert a.shock_corr == {} and "corr_e1_e2" not in a.params.names


def test_correlations_kept_positive_definite():
    three = Model([Equation("a = e1"), Equation("b = e2"), Equation("c = e3")])
    sol = three.linearize(shocks=["e1", "e2", "e3"]).solve()
    bad = {("e1", "e2"): 0.9, ("e2", "e3"): 0.9, ("e1", "e3"): -0.9}
    with pytest.raises(ModelError, match="not positive definite"):
        sol.system(["a", "b", "c"], shock_corr=bad)
    from pymetrix.models.estimation import _PENALTY

    ar = Model([Equation("a ~ r*a[-1] + e1", params=["r"]), Equation("b = e2"), Equation("c = e3")], params={"r": 0.5})
    shocks = ["e1", "e2", "e3"]
    start = {("e1", "e2"): 0.3, ("e2", "e3"): 0.3, ("e1", "e3"): 0.3}
    frame = ar.linearize(shocks=shocks).solve().simulate(120, {s: 1.0 for s in shocks}, seed=2, shock_corr=start)
    frame.index = _quarters(120)
    fit = ar.estimate(Dataset.from_frame(frame), ["a", "b", "c"], shocks=shocks, shock_corr=start,
                      estimate=["corr_e1_e2", "corr_e1_e3", "corr_e2_e3"], start="stationary")
    lik = fit.evaluate.__self__  # the likelihood behind the fit
    assert lik.free == ("corr_e1_e2", "corr_e1_e3", "corr_e2_e3")
    assert lik.objective(np.array([0.3, 0.3, 0.3])) < _PENALTY
    assert lik.objective(np.array([0.9, -0.9, 0.9])) == _PENALTY  # each within (-1, 1), together not a correlation matrix
    assert lik.objective(np.array([1.5, 0.0, 0.0])) == _PENALTY  # outside (-1, 1)
    assert lik.limits["corr_e1_e2"] == (-1.0, 1.0)


def test_refusals():
    with pytest.raises(ModelError, match="eq, which is not a shock"):
        read_mod(NK.replace("var ei; stderr 0.2; end;", "var ei; stderr 0.2; corr ey, eq = 0.4; end;"))
    with pytest.raises(ModelError, match="outside \\(-1, 1\\)"):
        read_mod(NK.replace("var ei; stderr 0.2; end;", "var ei; stderr 0.2; corr ey, ei = 1.2; end;"))
    with pytest.raises(ModelError, match="outside \\(-1, 1\\)"):
        read_mod(NK.replace("var ei; stderr 0.2; end;", "var ei; stderr 0.2; var ey, ei = 0.5; end;"))  # 0.5 / (0.5 * 0.2) = 5
    with pytest.raises(ModelError, match="two different shocks"):
        read_mod(NK.replace("var ei; stderr 0.2; end;", "var ei; stderr 0.2; corr ei, ei = 0.4; end;"))
    with pytest.raises(ModelError, match="not a shock"):
        read_mod(TWO + "estimated_params; corr e1, e9, 0, -1, 1; end;")
    sol = read_mod(NK).solve()
    with pytest.raises(ModelError, match="not a shock"):
        sol.irf(4, shock_corr={("ey", "ez"): 0.1})
    with pytest.raises(ModelError, match="outside"):
        sol.moments(shock_corr={("ey", "ei"): -1.0})
    with pytest.raises(ModelError, match="given twice"):
        sol.system(["y"], shock_corr={("ey", "ei"): 0.1, ("ei", "ey"): 0.2})
    with pytest.raises(ModelError, match="corr_<shock>_<shock>"):
        sol.system(["y"], shock_corr={"corr_ey_eq": 0.1})
    assert sol.system(["y"], shock_corr={"corr_ey_ei": 0.1}).shock_corr == {("ey", "ei"): 0.1}  # by name too


def test_the_dynare_writer_writes_corr():
    text = TWO + "estimated_params;\ncorr e2, e1, 0.1, -0.5, 0.9;\nstderr e1, inv_gamma_pdf, 0.5, inf;\nend;\n"
    dyn = read_mod(text)
    assert dyn.estimated == ("corr_e1_e2", "sd_e1") and dyn.bounds["corr_e1_e2"] == (-0.5, 0.9) and dyn.start["corr_e1_e2"] == 0.1
    out = export_model(dyn, "dynare")
    assert "corr e1, e2 = 0.5;" in out.text and "corr e1, e2, 0.1, -0.5, 0.9;" in out.text
    back = read_mod(out.text)
    assert back.shock_corr == dyn.shock_corr and back.bounds["corr_e1_e2"] == (-0.5, 0.9) and out.checked
    written = export_model(dyn, "pymetrix")  # corr a, b = r in pymetrix's shocks section, read back the same
    assert "    corr e1, e2 = 0.5" in written.text and written.checked
    assert ModelSource(written.text).build().shock_corr == dyn.shock_corr
    nk = templates.new_keynesian()
    nk = type(nk)(**{**nk.__dict__, "shock_corr": {("ey", "ei"): 0.25}})
    assert "corr ey, ei = 0.25;" in export_model(nk, "dynare").text
    assert nk.system().shock_corr == {("ey", "ei"): 0.25}
