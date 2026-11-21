"""DSGE-VAR(lambda) (Del Negro and Schorfheide 2004): the prior's population moments, the posterior's limits in lambda,
the closed-form marginal likelihood, the model's rotation, lambda-hat and the sampler.

Checked independently: the moments against a long simulation of the model (through its measurement equations); the
marginal likelihood against the normal-inverse-Wishart identity p(Y) = p(Y | Phi, Sigma) p(Phi, Sigma) / p(Phi, Sigma | Y)
at arbitrary points, each density from scipy; the responses at a very large lambda against the model's own, for a model
whose VAR(1) representation is exact."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pymetrix import Dataset, Equation, Model, SpecificationError, templates
from pymetrix.estimation.priors import Beta, Gamma, Normal
from pymetrix.metadata import Role
from pymetrix.models.dsge_var import _lag_design, population_moments, rotation, var_approximation, var_posterior


def _quarters(n, start="1960Q1"):
    return pd.period_range(start, periods=n, freq="Q", name="time")


def _two():
    """A fully observed model that is exactly a VAR(1), with a non-triangular impact of its two shocks."""
    return Model([Equation("x ~ a*x[-1] + b*z[-1] + ex + h*ez", params=["a", "b", "h"]),
                  Equation("z ~ c*z[-1] + g*ex + ez", params=["c", "g"])],
                 params={"a": 0.6, "b": 0.2, "h": 0.4, "c": 0.5, "g": -0.3}, roles={"ex": Role.SHOCK, "ez": Role.SHOCK})


@pytest.fixture(scope="module")
def two_data():
    rng = np.random.default_rng(4)
    x = np.zeros((161, 2))
    for t in range(1, 161):
        ex, ez = rng.normal(size=2) * [0.5, 0.8]
        x[t, 0] = 0.6 * x[t - 1, 0] + 0.2 * x[t - 1, 1] + ex + 0.4 * ez
        x[t, 1] = 0.5 * x[t - 1, 1] - 0.3 * ex + ez
    return Dataset.from_frame(pd.DataFrame(x[1:], columns=["x", "z"], index=_quarters(160)))


@pytest.fixture(scope="module")
def nk_data():
    nk = templates.new_keynesian()
    truth = nk.solve().simulate(160, nk.shock_sd, seed=1)
    truth.index = _quarters(160, "1980Q1")
    return nk, nk.measurement.apply(Dataset.from_frame(truth))


def _nk_fit(nk, data, **kw):
    kw.setdefault("lambda_", 1.0)
    return nk.model.dsge_var(data, nk.measurement, lags=2, shocks=nk.shocks, shock_sd=nk.shock_sd, estimate=["kappa", "rho"],
                             priors={"kappa": Gamma(0.2, 0.1), "rho": Beta(0.7, 0.1)}, **kw)


def test_population_moments_match_a_long_simulation():
    nk = templates.new_keynesian()
    p, n_sim = 2, 200_000
    system = nk.system()
    G_yy, G_xy, G_xx = population_moments(system, p)
    sim = nk.solve().simulate(n_sim, nk.shock_sd, seed=11)
    sim.index = _quarters(len(sim), "1000Q1")
    observed = nk.measurement.observed(nk.measurement.apply(Dataset.from_frame(sim))).dropna().to_numpy()
    Y, X = _lag_design(observed, p)
    T = len(Y)
    dx, dy = np.diag(G_xx), np.diag(G_yy)
    for population, sample, rows, cols in ((G_yy, Y.T @ Y / T, dy, dy), (G_xy, X.T @ Y / T, dx, dy), (G_xx, X.T @ X / T, dx, dx)):
        # each moment's error relative to its Cauchy-Schwarz bound; Monte Carlo error at 200,000 periods is under 0.01
        assert np.max(np.abs(sample - population) / np.sqrt(np.outer(rows, cols))) < 0.02
    # the constant's row is the means; the centred variances are the model's own (Moments, through the measurement)
    assert np.allclose(G_xy[0], system.d) and G_xx[0, 0] == 1
    assert np.allclose(G_xx[1:4, 1:4], G_yy) and np.allclose(G_xx[4:7, 4:7], G_yy)


def test_the_posterior_moves_from_least_squares_to_the_model_as_lambda_grows(two_data):
    fit = _two().dsge_var(two_data, ["x", "z"], lags=1, shock_sd={"ex": 0.5, "ez": 0.8}, estimate=["a"], lambda_=1.0)
    Y, X = fit._design()
    _, _, system = fit.fit.evaluate(fit.fit.params.as_dict())
    moments = population_moments(system, 1)
    phi_star, sigma_star = var_approximation(*moments)
    ols = np.linalg.lstsq(X, Y, rcond=None)[0]
    lambdas = [fit.lambda_min, 0.1, 0.5, 1, 5, 50, 1e3, 1e7]
    posts = [var_posterior(*moments, Y, X, lam) for lam in lambdas]
    to_model = [np.linalg.norm(p.phi - phi_star) for p in posts]
    to_ols = [np.linalg.norm(p.phi - ols) for p in posts]
    assert all(np.diff(to_model) < 0) and all(np.diff(to_ols) > 0)  # monotone in lambda, both ways
    assert to_model[-1] < 1e-6 and np.allclose(posts[-1].sigma, sigma_star, atol=1e-5)  # lambda -> infinity: the model's VAR
    assert to_ols[0] < 0.5 * to_model[0]  # at the smallest lambda, nearer least squares than the model
    T = len(Y)
    for lam, p in zip(lambdas[:4], posts):  # the posterior mean is the precision-weighted average of the two
        weighted = np.linalg.solve(lam * T * moments[2] + X.T @ X, lam * T * moments[2] @ phi_star + X.T @ X @ ols)
        assert np.allclose(p.phi, weighted, atol=1e-10)
    with pytest.raises(SpecificationError, match="at least"):
        var_posterior(*moments, Y, X, 0.5 * fit.lambda_min)
    with pytest.raises(SpecificationError, match="at least"):
        _two().dsge_var(two_data, ["x", "z"], lags=1, estimate=["a"], lambda_=0.01)


def _log_identity(moments, Y, X, lam, phi, sigma):
    """log p(Y) by p(Y | Phi, Sigma) p(Phi, Sigma) / p(Phi, Sigma | Y) at (phi, sigma), each density from scipy."""
    T, n = Y.shape
    k = X.shape[1]
    phi_star, sigma_star = var_approximation(*moments)
    G_xx = moments[2]
    post = var_posterior(*moments, Y, X, lam)
    like = stats.multivariate_normal(np.zeros(n), sigma).logpdf(Y - X @ phi).sum()

    def matrix_normal(mean, row_cov):  # vec(Phi) ~ N(vec(mean), Sigma (x) row_cov), Phi k x n
        return stats.matrix_normal(mean, row_cov, sigma).logpdf(phi)

    prior = stats.invwishart(df=lam * T - k, scale=lam * T * sigma_star).logpdf(sigma) + matrix_normal(phi_star, np.linalg.inv(lam * T * G_xx))
    posterior = stats.invwishart(df=post.df, scale=post.scale).logpdf(sigma) + matrix_normal(post.phi, np.linalg.inv(post.precision))
    return like + prior - posterior, post.logml


def test_the_closed_form_marginal_likelihood_equals_the_identity(nk_data):
    nk, data = nk_data
    fit = _nk_fit(nk, data)
    Y, X = fit._design()
    rng = np.random.default_rng(0)
    for values in (fit.fit.params.as_dict(), {"kappa": 0.3, "rho": 0.6}):
        _, _, system = fit.fit.evaluate(values)
        moments = population_moments(system, fit.lags)
        for lam in (fit.lambda_min * 1.01, 0.4, 2.5):
            post = var_posterior(*moments, Y, X, lam)
            for phi, sigma in ((post.phi, post.sigma),  # at the posterior mean, and at a point far from it
                               (post.phi + 0.05 * rng.normal(size=post.phi.shape), 1.7 * post.sigma + 0.1 * np.eye(3))):
                direct, closed = _log_identity(moments, Y, X, lam, phi, sigma)
                assert closed == pytest.approx(direct, abs=1e-7)
    assert fit.fit.stats["loglike"] == pytest.approx(var_posterior(*population_moments(fit.fit.system, 2), Y, X, 1.0).logml)


def test_responses_at_a_large_lambda_are_the_models_own(two_data):
    fit = _two().dsge_var(two_data, ["x", "z"], lags=1, shock_sd={"ex": 0.5, "ez": 0.8}, estimate=["a", "c"],
                          priors={"a": Normal(0.5, 0.2), "c": Normal(0.5, 0.2)}, lambda_=1e6)
    model, var = fit.dsge_irf(12).values, fit.irf(12, draws=0).values
    assert np.max(np.abs(var - model)) < 1e-4 * np.max(np.abs(model))
    banded = fit.irf(12, draws=200, seed=1)
    assert np.max(np.abs(banded.upper - banded.lower)) < 0.02 * np.max(np.abs(model))  # little left to the data at lambda T = 1.6e8
    impact = fit.dsge_irf(0).values[0]
    omega = rotation(impact)
    assert np.allclose(omega @ omega.T, np.eye(2)) and np.allclose(np.triu(impact @ omega.T, 1), 0)
    # at a small lambda the VAR's responses part from the model's; the impact's rotation stays the model's
    loose = _two().dsge_var(two_data, ["x", "z"], lags=1, shock_sd={"ex": 0.5, "ez": 0.8}, estimate=["a"], lambda_=0.1)
    assert np.max(np.abs(loose.irf(12, draws=0).values - loose.dsge_irf(12).values)) > 1e-3
    fevd = fit.fevd(8).values
    assert np.allclose(fevd.sum(axis=2), 1)


def test_lambda_hat_on_a_grid_and_lambda_estimated(nk_data):
    nk, data = nk_data
    grid = _nk_fit(nk, data, lambda_="grid", lambdas=[0.25, 0.5, 1, 2])
    table = grid.grid
    assert list(table.index) == [0.25, 0.5, 1, 2] and grid.lambda_choice == "grid"
    assert grid.lambda_ == table["log marginal density"].idxmax()
    assert grid.fit.stats["logpost"] == pytest.approx(table.loc[grid.lambda_, "log posterior"])
    assert "lambda-hat" in str(grid.lambdas()) and grid.lambda_plot().panels
    with pytest.raises(SpecificationError, match="below"):
        _nk_fit(nk, data, lambda_="grid", lambdas=[0.01, 1])
    est = _nk_fit(nk, data, lambda_=Gamma(1.0, 0.5))
    assert "lambda" in est.fit.params.names and est.lambda_ == est.fit.params.as_dict()["lambda"] > est.lambda_min
    with pytest.raises(SpecificationError):
        _nk_fit(nk, data, lambda_="best")


def test_the_posterior_of_theta_and_what_it_gives(nk_data):
    nk, data = nk_data
    fit = _nk_fit(nk, data).mcmc(draws=200, chains=1, seed=3)
    post = fit.posterior
    assert post.draws.shape == (1, 200, 2) and fit.provenance.seed == 3
    assert fit.provenance.method["lambda choice"] == "fixed" and fit.provenance.data_fingerprint
    B, S = fit.var_draws(50, seed=2)
    assert B.shape == (50, 7, 3) and S.shape == (50, 3, 3) and np.all(np.linalg.eigvalsh(S) > 0)
    B2, _ = fit.var_draws(50, seed=2)
    assert np.array_equal(B, B2)  # the same seed, the same draws
    irf = fit.irf(8, draws=50, seed=1)
    assert irf.lower.shape == irf.values.shape and np.all(irf.lower <= irf.upper)
    forecast = fit.forecast(("2020Q1", "2020Q4"), draws=50, seed=1)
    assert list(forecast.frame.columns[:5]) == ["infl", "infl_se", "infl_median", "infl_lower", "infl_upper"]
    assert len(forecast.frame) == 4 and np.all(forecast.frame["infl_lower"] < forecast.frame["infl_upper"])
    assert "HPD" in str(fit.summary()) and "modified harmonic mean" in str(fit.summary())


DSGE_VAR_MOD = """
var y;
varexo e;
parameters rho dsge_prior_weight;
rho = 0.5; dsge_prior_weight = 1;
model;
y = rho*y(-1) + e;
end;
shocks; var e; stderr 0.2; end;
estimated_params;
rho, beta_pdf, 0.6, 0.1;
stderr e, inv_gamma_pdf, 0.1, inf;
dsge_prior_weight, uniform_pdf, , , 0, 2;
end;
varobs y;
estimation(datafile=ardata, dsge_var, dsge_varlag=2, mh_replic=0);
"""


def test_dynare_dsge_var_options(tmp_path):
    """Dynare's estimation(dsge_var, dsge_varlag=2) with dsge_prior_weight in estimated_params: lambda estimated with its
    prior, as Model.dsge_var does by hand; dsge_var=<value> fixes it. (No Dynare outputs of dsge_var are recorded here.)"""
    from pymetrix import MeasurementMap, read_mod
    from pymetrix.estimation.priors import InvGamma1, Uniform

    rng = np.random.default_rng(3)
    x = np.zeros(120)
    for t in range(1, 120):
        x[t] = 0.7 * x[t - 1] + 0.2 * rng.normal()
    (tmp_path / "ardata.m").write_text("y = [" + "\n".join(f"{v:.12g}" for v in x) + "];\n", encoding="utf-8")
    path = tmp_path / "ar.mod"
    path.write_text(DSGE_VAR_MOD, encoding="utf-8")
    dyn = read_mod(path)
    fit = dyn.dsge_var()
    assert fit.lags == 2 and fit.lambda_choice == "estimated" and fit.fit.priors["lambda"] == Uniform(0, 2)
    data = dyn.data()
    by_hand = Model([Equation("y ~ rho*y[-1] + e", params=["rho"])], params={"rho": 0.5}).dsge_var(
        data, MeasurementMap.direct(["y"]), lags=2, lambda_=Uniform(0, 2), shocks=["e"], shock_sd={"e": 0.2},
        estimate=["rho", "sd_e"], priors={"rho": Beta(0.6, 0.1), "sd_e": InvGamma1(0.1, np.inf)})
    assert fit.fit.stats["logpost"] == pytest.approx(by_hand.fit.stats["logpost"], abs=1e-6)
    assert np.allclose(fit.fit.params.values, by_hand.fit.params.values, atol=1e-4)
    path.write_text(DSGE_VAR_MOD.replace("dsge_var,", "dsge_var=0.5,").replace("dsge_prior_weight, uniform_pdf, , , 0, 2;\n", ""),
                    encoding="utf-8")
    fixed = read_mod(path).dsge_var()
    assert fixed.lambda_ == 0.5 and fixed.lambda_choice == "fixed" and "lambda" not in fixed.fit.params.names


def test_the_marginal_density_humps_when_the_model_is_somewhat_wrong():
    """Data from a model with another intertemporal elasticity: the restrictions help, but not without limit, so the
    marginal data density rises from the smallest lambda and falls after lambda-hat (DS 2004's hump)."""
    nk = templates.new_keynesian()
    sim = templates.new_keynesian(sigma=2).solve().simulate(120, nk.shock_sd, seed=5)
    sim.index = _quarters(120, "1990Q1")
    dv = nk.dsge_var(nk.measurement.apply(Dataset.from_frame(sim)), lags=1, lambda_="grid", lambdas=[0.1, 0.25, 0.5, 1, 2, 5],
                     estimate=["rho", "kappa"], priors={"rho": Beta(0.7, 0.1), "kappa": Gamma(0.15, 0.05)})
    density = dv.grid["log marginal density"].to_numpy()
    top = int(np.argmax(density))
    assert 0 < top < len(density) - 1 and dv.lambda_ == dv.grid.index[top]
    assert np.all(np.diff(density[:top + 1]) > 0) and np.all(np.diff(density[top:]) < 0)


def test_the_dialog_and_metropolis_hastings_from_the_app(nk_data):
    """Model > Estimate DSGE-VAR on a template, then Model > Metropolis-Hastings on the result; the script says the same."""
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import EstimateDSGEVAR, RunMCMC

    nk, data = nk_data
    s = Session()
    s.objects["nk"], s.objects["data"] = nk, data
    s.run(EstimateDSGEVAR("nk", lags=2, lambda_="Gamma(1, 0.5)", estimate="kappa, rho", priors="kappa=Gamma(0.2, 0.1); rho=Beta(0.7, 0.1)",
                          name="dv"))
    direct = _nk_fit(nk, data, lambda_=Gamma(1.0, 0.5))
    assert s["dv"].fit.stats["logpost"] == pytest.approx(direct.fit.stats["logpost"], abs=1e-9)
    s.run(RunMCMC("dv", draws=30, chains=1, burn=20, workers=1, name="post"))
    assert s["post"].posterior.draws.shape == (1, 30, 3)
    script = s.script()
    assert "dv = nk.dsge_var(data, lags=2, lambda_=priors.Gamma(mean=1.0, sd=0.5), estimate=['kappa', 'rho'], priors=" in script
    assert "post = dv.mcmc(draws=30, chains=1, burn=20, seed=1, workers=1)" in script
    grid = EstimateDSGEVAR("nk", lags=1, lambdas="0.5, 1", estimate="rho", priors="rho=Beta(0.7, 0.1)", name="g")
    assert grid.to_code() == "g = nk.dsge_var(data, lags=1, lambda_='grid', lambdas=[0.5, 1.0], estimate=['rho'], priors={'rho': priors.Beta(0.7, 0.1)})"
    assert EstimateDSGEVAR("nk", lambda_="Nope(1)").validate(s)
