"""The analytic gradient of a linearised model's Kalman-filter log likelihood (pymetrix.models.kalman_score).

Checked against five-point central differences of the log likelihood as the Kalman filter computes it
(``_Likelihood.loglike``, the statsmodels or native filter: an implementation apart from the differentiated one), to
1e-6 relative, on: the New Keynesian template with measurement errors and correlated shocks (every kind of parameter:
structural, shock sizes, a correlation, measurement errors); a cash-in-advance model in Dynare's syntax (Schorfheide
2000, as in Dynare's fs2000 example, read by ``read_mod``: log-linearised, a nonlinear steady state, leads of two
periods); the growth model with nonlinear measurement equations (logs); a model with a lag of two periods, a static
variable and a lagged measurement equation, with a presample, missing observations and the scaled start. Then: the
posterior mode and standard errors with the analytic gradient agree with the finite-difference ones; NUTS takes the
analytic gradient; where it does not apply (regimes, a particle filter) the finite differences are kept; the dialogs.
The filter's adjoint on its own, on a random system whose every matrix moves, against five-point differences of a
textbook filter (an implementation apart); NUTS chains in other processes give the same draws. Smets and Wouters
(2007), in tests/test_dynare_sw2007.py.
"""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, Model, read_mod, templates
from pymetrix.estimation.priors import Beta, Gamma, InvGamma1, Normal
from pymetrix.metadata import Role
from pymetrix.models.kalman_score import loglike_score, ready
from pymetrix.models.measurement import MeasurementMap
from pymetrix.models.solvers import ModelError


def _periods(n, start="1990Q1"):
    return pd.period_range(start, periods=n, freq="Q", name="time")


def _lik(fit):
    return fit.log_posterior.__self__


def _five_point(lik, point):
    """d loglike / d each parameter by the five-point stencil (error of order h^4)."""
    names = list(lik.free)
    out = []
    for n in names:
        h = 1e-4 * max(abs(point[n]), 1e-3)  # small against each parameter's scale: the error is (h / scale)^4
        f = [lik.loglike({**point, n: point[n] + k * h}) for k in (-2, -1, 1, 2)]
        out.append((f[0] - 8 * f[1] + 8 * f[2] - f[3]) / (12 * h))
    return np.array(out)


def _check(fit, point, rtol=1e-6):
    lik = _lik(fit)
    ll, g = loglike_score(lik, point)
    assert ll == pytest.approx(lik.loglike(point), rel=1e-9)  # the value is the filter's (statsmodels' to its convergence tolerance)
    ref = _five_point(lik, point)
    tol = rtol * np.maximum(np.abs(ref), 1e-3 * np.max(np.abs(ref)))
    assert np.all(np.abs(g - ref) <= tol), dict(zip(lik.free, (g - ref) / np.abs(ref)))
    return g


@pytest.fixture(scope="module")
def nk():
    nk = templates.new_keynesian()
    corr = {("ey", "ei"): 0.3}
    m = nk.measurement.with_errors({o: 0.3 for o in nk.measurement.observables})
    path = nk.solve().simulate(100, nk.shock_sd, seed=11)
    path.index = _periods(100)
    frame = m.apply(Dataset.from_frame(path)).frame + 0.3 * np.random.default_rng(0).standard_normal((100, 3))
    data = Dataset.from_frame(frame)
    names = ["sigma", "kappa", "rho", "phi_pi", "phi_y", "sd_ey", "sd_epi", "sd_ei", "corr_ey_ei", "sd_infl", "sd_growth"]
    pri = {"kappa": Gamma(0.1, 0.05), "rho": Beta(0.7, 0.1), "phi_pi": Normal(1.5, 0.25), "phi_y": Gamma(0.5, 0.2),
           "sigma": Gamma(1.0, 0.3), **{f"sd_{s}": InvGamma1(0.3, 1.0) for s in ("ey", "epi", "ei", "infl", "growth")},
           "corr_ey_ei": Beta(0.5, 0.2)}
    fit = nk.estimate(data, names, priors=pri, measurement=m, shock_corr=corr, cov="none", max_iter=3)
    return nk, m, data, fit, corr, pri


def test_the_new_keynesian_template_with_errors_and_correlated_shocks(nk):
    _, _, _, fit, _, _ = nk
    point = {"sigma": 1.3, "kappa": 0.15, "rho": 0.7, "phi_pi": 1.7, "phi_y": 0.3, "sd_ey": 0.4, "sd_epi": 0.35,
             "sd_ei": 0.25, "corr_ey_ei": 0.4, "sd_infl": 0.4, "sd_growth": 0.25}
    g = _check(fit, point)
    frame = fit.gradient(at=point)
    assert np.allclose(frame["d log likelihood"], g) and frame.attrs["method"] == "analytic"
    lik = _lik(fit)
    x = np.array([point[n] for n in lik.free])
    prior_part = frame["d log posterior"] - frame["d log likelihood"]
    for i, n in enumerate(lik.free):  # the priors' part: each prior's own derivative
        h = 1e-5 * max(1.0, abs(x[i]))
        assert prior_part.iloc[i] == pytest.approx((lik.priors[n].logpdf(x[i] + h) - lik.priors[n].logpdf(x[i] - h)) / (2 * h), rel=1e-6)
    numerical = fit.gradient(at=point, method="numerical")
    assert np.allclose(numerical["d log likelihood"], g, rtol=1e-4)


FS = """
var m P c e W R k d n l gy_obs gp_obs y dA;
varexo e_a e_m;
parameters alp bet gam mst rho psi del;
alp = 0.33; bet = 0.99; gam = 0.003; mst = 1.011; rho = 0.7; psi = 0.787; del = 0.02;
model;
dA = exp(gam+e_a);
log(m) = (1-rho)*log(mst) + rho*log(m(-1))+e_m;
-P/(c(+1)*P(+1)*m)+bet*P(+1)*(alp*exp(-alp*(gam+log(e(+1))))*k^(alp-1)*n(+1)^(1-alp)+(1-del)*exp(-(gam+log(e(+1)))))/(c(+2)*P(+2)*m(+1))=0;
W = l/n;
-(psi/(1-psi))*(c*P/(1-n))+l/n = 0;
R = P*(1-alp)*exp(-alp*(gam+e_a))*k(-1)^alp*n^(-alp)/W;
1/(c*P)-bet*P*(1-alp)*exp(-alp*(gam+e_a))*k(-1)^alp*n^(1-alp)/(m*l*c(+1)*P(+1)) = 0;
c+k = exp(-alp*(gam+e_a))*k(-1)^alp*n^(1-alp)+(1-del)*exp(-(gam+e_a))*k(-1);
P*c = m;
m-1+d = l;
e = exp(e_a);
y = k(-1)^alp*n^(1-alp)*exp(-alp*(gam+e_a));
gy_obs = dA*y/y(-1);
gp_obs = (P/P(-1))*m(-1)/dA;
end;
initval;
k = 6; m = 1.011; P = 2.25; c = 0.45; e = 1; W = 4; R = 1.02; d = 0.85; n = 0.19; l = 0.86; y = 0.6; gy_obs = 1.003;
gp_obs = 1.008; dA = 1.003;
end;
shocks; var e_a; stderr 0.014; var e_m; stderr 0.005; end;
estimated_params;
alp, beta_pdf, 0.356, 0.02;
bet, beta_pdf, 0.993, 0.002;
gam, normal_pdf, 0.0085, 0.003;
mst, normal_pdf, 1.0002, 0.007;
rho, beta_pdf, 0.129, 0.223;
psi, beta_pdf, 0.65, 0.05;
del, beta_pdf, 0.01, 0.005;
stderr e_a, inv_gamma_pdf, 0.035449, inf;
stderr e_m, inv_gamma_pdf, 0.008862, inf;
end;
varobs gp_obs gy_obs;
estimation(order=1, loglinear, mh_replic=0);
"""


@pytest.fixture(scope="module")
def fs():
    """Schorfheide's (2000) cash-in-advance model as Dynare's fs2000 example writes it (the equations; parameter values,
    priors and data here are this test's own: data simulated from the model)."""
    dyn = read_mod(FS)
    sim = dyn.solve().simulate(120, dyn.shock_sd, seed=3)
    sim.index = _periods(120, "1960Q1")
    data = Dataset.from_frame(sim[["gp_obs", "gy_obs"]])
    return dyn, data, dyn.estimate(data, cov="none", max_iter=2)


def test_a_dynare_cash_in_advance_model(fs):
    dyn, _, fit = fs
    assert dyn.loglinear and any(v.endswith("_lead1") for v in fit.solution.variables)  # leads of two periods: auxiliaries
    point = {"alp": 0.34, "bet": 0.99, "gam": 0.004, "mst": 1.01, "rho": 0.6, "psi": 0.7, "del": 0.015, "sd_e_a": 0.015,
             "sd_e_m": 0.006}
    _check(fit, point)


def test_nonlinear_measurement_equations_and_a_nonlinear_steady_state():
    rbc = templates.rbc()
    m = MeasurementMap(["log_y = log(y)", "log_c = log(c)", "dk = k - k[-1]"], errors={"log_y": 0.002, "log_c": 0.003, "dk": 0.01})
    sim = rbc.solve().simulate(80, rbc.shock_sd, seed=2)
    sim.index = _periods(80)
    data = Dataset.from_frame(m.apply(Dataset.from_frame(sim)).frame)
    fit = rbc.estimate(data, ["alpha", "beta", "delta", "sd_e", "sd_log_c"], measurement=m, cov="none", max_iter=1)
    _check(fit, {"alpha": 0.35, "beta": 0.985, "delta": 0.03, "sd_e": 0.012, "sd_log_c": 0.004})


@pytest.fixture(scope="module")
def lags():
    """A lag of two periods (an auxiliary state), a static variable, a forward-looking one, a lagged measurement."""
    model = Model([
        Equation("x ~ a*x[-1] + b*x[-2] + ex", params=["a", "b"]),
        Equation("z ~ beta*z[+1] + kappa*x + ez", params=["beta", "kappa"]),
        Equation("w ~ c*x + z", params=["c"]),
    ], params={"a": 0.6, "b": 0.2, "beta": 0.9, "kappa": 0.5, "c": 2.0}, roles={"ex": Role.SHOCK, "ez": Role.SHOCK})
    m = MeasurementMap(["ow = w", "dz = z - z[-1]"], errors={"ow": 0.2, "dz": 0.3})
    sol = model.linearize(model.steady_state(), shocks=["ex", "ez"]).solve()
    sim = sol.simulate(90, {"ex": 0.5, "ez": 0.4}, seed=5)
    sim.index = _periods(90)
    frame = m.apply(Dataset.from_frame(sim)).frame + 0.2 * np.random.default_rng(1).standard_normal((90, 2))
    frame.iloc[[10, 11, 40], 0] = np.nan  # missing observations: the full recursion in those periods
    frame.iloc[60, 1] = np.nan
    return model, m, Dataset.from_frame(frame)


@pytest.mark.parametrize("start, presample", [("auto", 0), ("stationary", 4), ("scaled", 2)])
def test_lags_statics_missing_data_presample_and_starts(lags, start, presample):
    model, m, data = lags
    fit = model.estimate(data, m, shocks=["ex", "ez"], shock_sd={"ex": 0.5, "ez": 0.4},
                         estimate=["a", "b", "beta", "kappa", "c", "sd_ex", "sd_ez", "sd_ow"], start=start, presample=presample,
                         cov="none", max_iter=1)
    point = {"a": 0.5, "b": 0.25, "beta": 0.85, "kappa": 0.6, "c": 1.5, "sd_ex": 0.6, "sd_ez": 0.3, "sd_ow": 0.25}
    _check(fit, point)


def test_the_mode_and_hessian_agree_with_finite_differences(nk):
    template, m, data, _, corr, pri = nk
    names = ["kappa", "rho", "phi_pi", "sd_ey", "sd_ei", "sd_infl"]
    chosen = {n: pri[n] for n in names}
    fast = template.estimate(data, names, priors=chosen, measurement=m, shock_corr=corr)
    slow = template.estimate(data, names, priors=chosen, measurement=m, shock_corr=corr, gradient="numerical")
    assert fast.provenance.method["gradient"].startswith("analytic")
    assert slow.provenance.method["gradient"] == "central differences (asked for)"
    assert fast.stats["logpost"] >= slow.stats["logpost"] - 1e-6
    assert np.allclose(fast.params.values, slow.params.values, rtol=1e-3, atol=1e-5)
    assert np.allclose(fast.params.se, slow.params.se, rtol=0.02)
    # the Hessian from the analytic gradient is the more accurate one: a five-point Hessian of the log posterior agrees
    lik = _lik(fast)
    x = fast.params.values
    h = 1e-3 * np.maximum(np.abs(x), 0.1)
    H = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            ei, ej = np.eye(len(x))[i] * h[i], np.eye(len(x))[j] * h[j]
            H[i, j] = sum(s * t * lik.log_posterior(x + s * ei + t * ej) for s in (1, -1) for t in (1, -1)) / (4 * h[i] * h[j])
    assert np.allclose(np.sqrt(np.diag(np.linalg.inv(-H))), fast.params.se, rtol=2e-3)


def test_nuts_takes_the_analytic_gradient(nk):
    template, m, data, _, corr, pri = nk
    names = ["kappa", "rho", "sd_ey"]
    fit = template.estimate(data, names, priors={n: pri[n] for n in names}, measurement=m, shock_corr=corr)
    post = fit.mcmc(draws=25, chains=1, burn=25, seed=2, sampler="nuts")
    assert post.provenance.method["gradient"] == "analytic (Kalman-filter score)"
    assert np.allclose(post.logpost.ravel(), [fit.log_posterior(row) for row in post.draws[0]], atol=1e-6)
    numerical = fit.mcmc(draws=10, chains=1, burn=10, seed=2, sampler="nuts", gradient="numerical")
    assert numerical.provenance.method["gradient"] == "central differences"
    with pytest.raises(Exception, match="gradient is one of"):
        fit.mcmc(draws=10, sampler="nuts", gradient="exact")


def test_where_it_does_not_apply_finite_differences_are_kept(nk):
    template, m, data, _, _, _ = nk
    fit = template.estimate(data, ["kappa", "rho", "kappa@2005Q1"], measurement=m, regimes="2005Q1: kappa=0.12",
                            cov="none", max_iter=2)
    lik = _lik(fit)
    assert not ready(lik, fit.params.values)
    assert fit.provenance.method["gradient"].startswith("central differences (known-date regimes")
    with pytest.raises(ModelError, match="no analytic gradient: known-date regimes"):
        fit.gradient()
    assert fit.gradient(method="numerical").shape == (3, 3)
    with pytest.raises(ModelError, match="no analytic gradient"):
        template.estimate(data, ["kappa"], measurement=m, regimes="2005Q1: kappa=0.12", gradient="analytic")


def test_the_dialogs(nk):
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import EstimateModel, RunMCMC
    from pymetrix.forms import form_spec

    _, _, data, _, _, _ = nk
    s = Session()
    s.objects.update({"nk": templates.new_keynesian(), "data": data})
    s.run(EstimateModel("nk", "data", estimate="kappa, rho", priors="kappa=Gamma(0.1, 0.05); rho=Beta(0.7, 0.1)",
                        gradient="numerical", name="fit"))
    api = templates.new_keynesian().estimate(data, ["kappa", "rho"], priors={"kappa": Gamma(0.1, 0.05), "rho": Beta(0.7, 0.1)},
                                            gradient="numerical")
    assert np.allclose(s["fit"].params.values, api.params.values)
    assert "gradient='numerical'" in s.history[-1].code
    assert s["fit"].provenance.method["gradient"] == "central differences (asked for)"
    s.run(RunMCMC("fit", draws=10, chains=1, seed=1, burn=10, sampler="nuts", gradient="analytic", name="post"))
    assert s.history[-1].code == "post = fit.mcmc(draws=10, chains=1, burn=10, seed=1, sampler='nuts', gradient='analytic')"
    direct = api.mcmc(draws=10, chains=1, seed=1, burn=10, sampler="nuts", gradient="analytic")
    assert np.allclose(s["post"].draws, direct.draws)
    namespace = dict(s.namespace) | {"nk": templates.new_keynesian(), "data": data}
    for step in s.history:  # the generated script replays
        exec(step.code, namespace)  # noqa: S102 - the session's own script
    assert np.allclose(namespace["post"].draws, s["post"].draws)
    assert {"gradient", "smooth_paths"} <= {f.name for f in form_spec(EstimateModel)}
    assert "gradient" in {f.name for f in form_spec(RunMCMC)}


# --- the filter's adjoint on its own, against a textbook filter --------------------------------------------------------

def _textbook_loglike(y, T, RQR, Z, d, H, P0, presample):
    """The Kalman filter as in any textbook (Harvey 1989): no converged-variance shortcut, missing observations by
    dropping their rows, numpy.linalg throughout. An implementation apart from the one differentiated."""
    a, P, total = np.zeros(T.shape[0]), P0, 0.0
    for t, row in enumerate(y):
        seen = ~np.isnan(row)
        if seen.any():
            Zs, Hs = Z[seen], H[np.ix_(seen, seen)]
            v = row[seen] - d[seen] - Zs @ a
            F = Zs @ P @ Zs.T + Hs
            Finv = np.linalg.inv(F)
            if t >= presample:
                total += -0.5 * (seen.sum() * np.log(2 * np.pi) + np.linalg.slogdet(F)[1] + v @ Finv @ v)
            K = P @ Zs.T @ Finv
            a, P = a + K @ v, P - K @ Zs @ P
        a, P = T @ a, T @ P @ T.T + RQR
    return total


@pytest.mark.parametrize("stationary", [True, False])
def test_the_adjoint_against_differences_of_a_textbook_filter(stationary):
    """A random stable system whose every matrix moves with five parameters (``T``, ``Z``, ``d``, ``H``, ``R Q R'`` and,
    with the stationary start, the initial variance through them), 300 periods (the variance converges and is frozen),
    missing observations before and after it converges (the frozen state left and taken up again), a presample."""
    from scipy import linalg

    from pymetrix.models.kalman_score import _score_filter

    rng = np.random.default_rng(3)
    n, p, K, n_t = 5, 3, 5, 300
    T0 = rng.standard_normal((n, n))
    T0 *= 0.8 / np.max(np.abs(np.linalg.eigvals(T0)))
    R0 = rng.standard_normal((n, 2))
    Z0, d0 = rng.standard_normal((p, n)), rng.standard_normal(p)
    dT = 0.05 * rng.standard_normal((K, n, n))
    dR = 0.1 * rng.standard_normal((K, n, 2))
    dZ, dd = 0.1 * rng.standard_normal((K, p, n)), rng.standard_normal((K, p))
    dh = 0.05 * rng.random((K, p))

    def system(theta):
        T = T0 + np.tensordot(theta, dT, 1)
        R = R0 + np.tensordot(theta, dR, 1)
        H = np.diag(0.3 + np.tensordot(theta, dh, 1))
        return T, R @ R.T, Z0 + np.tensordot(theta, dZ, 1), d0 + theta @ dd, H

    def start(T, RQR):
        return linalg.solve_discrete_lyapunov(T, RQR) if stationary else 5.0 * np.eye(n)

    y = rng.standard_normal((n_t, p))
    y[[3, 4], 0] = np.nan
    y[7] = np.nan
    y[250, [1, 2]] = np.nan  # after the variance has converged
    theta = 0.3 * rng.standard_normal(K)
    T, RQR, Z, d, H = system(theta)
    # the derivatives of the system matrices in each parameter
    Rt = R0 + np.tensordot(theta, dR, 1)
    dRQR = dR @ Rt.T + Rt @ np.swapaxes(dR, 1, 2)
    dH = np.array([np.diag(h) for h in dh])
    ll, g = _score_filter(y, T, RQR, Z, d, H, start(T, RQR), dT, dRQR, dZ, dd, dH, 2, stationary)
    assert ll == pytest.approx(_textbook_loglike(y, T, RQR, Z, d, H, start(T, RQR), 2), rel=1e-11)
    ref = []
    for k in range(K):
        h = 1e-4
        f = [_textbook_loglike(y, *system(theta + s * h * np.eye(K)[k])[:3], *system(theta + s * h * np.eye(K)[k])[3:],
                               start(*system(theta + s * h * np.eye(K)[k])[:2]), 2) for s in (-2, -1, 1, 2)]
        ref.append((f[0] - 8 * f[1] + 8 * f[2] - f[3]) / (12 * h))
    np.testing.assert_allclose(g, ref, rtol=1e-7)


def test_parallel_nuts_chains_give_the_same_draws(nk):
    """NUTS chains in other processes (workers=2) give the draws of chains run here, one after another; the dialog's
    Workers field writes the same call, and its script replays."""
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import RunMCMC

    template, m, data, _, corr, pri = nk
    names = ["kappa", "rho"]
    fit = template.estimate(data, names, priors={n: pri[n] for n in names}, measurement=m, shock_corr=corr)
    here = fit.mcmc(draws=10, chains=2, burn=10, seed=4, sampler="nuts", workers=1)
    assert not np.array_equal(here.draws[0], here.draws[1])  # each chain its own stream
    s = Session()
    s.objects["fit"] = fit
    s.run(RunMCMC("fit", draws=10, chains=2, seed=4, burn=10, workers=2, sampler="nuts", name="post"))
    assert s.history[-1].code == "post = fit.mcmc(draws=10, burn=10, seed=4, workers=2, sampler='nuts')"
    assert np.array_equal(s["post"].draws, here.draws)
    namespace = {"fit": fit}
    exec(s.history[-1].code, namespace)  # noqa: S102 - the session's own script
    assert np.array_equal(namespace["post"].draws, here.draws)
