"""Bayesian estimation of linearised models: Metropolis-Hastings, diagnostics, marginal likelihood, identification.

Checked against the posterior computed exactly on a grid (one parameter), effective sample sizes of AR(1) chains
with a known autocorrelation, and the analytic moments of an AR(1)."""

import importlib.util

import numpy as np
import pandas as pd
import pytest
from scipy import integrate

from pymetrix import Dataset, Equation, Model, templates
from pymetrix.estimation.priors import Gamma, Normal
from pymetrix.metadata import Role
from pymetrix.models import ModelError, Posterior
from pymetrix.models.bayes import ess, hpd, rhat


def _quarters(n: int) -> pd.PeriodIndex:
    return pd.period_range("1950Q1", periods=n, freq="Q", name="time")


@pytest.fixture(scope="module")
def ar1_fit():
    rng = np.random.default_rng(1)
    x = np.zeros(120)
    for t in range(1, 120):
        x[t] = 0.7 * x[t - 1] + 0.5 * rng.normal()
    data = Dataset.from_frame(pd.DataFrame({"x": x}, index=_quarters(120)))
    model = Model([Equation("x ~ rho*x[-1] + e", params=["rho"])], params={"rho": 0.3}, roles={"e": Role.SHOCK})
    return model, data, model.estimate(data, ["x"], estimate=["rho"], shock_sd={"e": 0.5}, priors={"rho": Normal(0.5, 0.2)})


@pytest.fixture(scope="module")
def ar1_posterior(ar1_fit):
    return ar1_fit[2].mcmc(draws=3000, chains=2, seed=5)


def _grid(fit):
    g = np.linspace(0.2, 1.0, 1601)
    lp = np.array([fit.log_posterior([v]) for v in g])
    top = lp[np.isfinite(lp)].max()
    w = np.exp(np.where(np.isfinite(lp), lp - top, -np.inf))
    z = integrate.trapezoid(w, g)
    mean = integrate.trapezoid(w * g, g) / z
    sd = np.sqrt(integrate.trapezoid(w * (g - mean) ** 2, g) / z)
    return mean, sd, top + np.log(z)


def test_the_posterior_matches_the_exact_one_on_a_grid(ar1_fit, ar1_posterior):
    fit, post = ar1_fit[2], ar1_posterior
    mean, sd, log_ml = _grid(fit)
    draws = post["rho"]
    mc_se = draws.std() / np.sqrt(ess(post.draws[:, :, 0]))
    assert abs(draws.mean() - mean) < 4 * mc_se
    assert draws.std() == pytest.approx(sd, rel=0.08)
    # 0.1: about three Monte Carlo standard deviations of the harmonic mean at 2 x 3000 draws (0.031 over 12 seeds)
    assert post.marginal_likelihood() == pytest.approx(log_ml, abs=0.1)
    assert post.laplace() == pytest.approx(log_ml, abs=0.05)
    assert all(0.15 < a < 0.5 for a in post.acceptance)
    assert rhat(post.draws[:, :, 0]) < 1.01


def test_diagnostics_against_known_answers():
    rng = np.random.default_rng(0)
    phi, n = 0.8, 20000
    chains = np.zeros((4, n))
    for c in range(4):
        e = rng.normal(size=n)
        for t in range(1, n):
            chains[c, t] = phi * chains[c, t - 1] + e[t]
    assert ess(chains) == pytest.approx(4 * n * (1 - phi) / (1 + phi), rel=0.1)  # AR(1): n (1 - phi) / (1 + phi)
    assert ess(rng.normal(size=(2, 5000))) == pytest.approx(10000, rel=0.1)
    assert rhat(chains) == pytest.approx(1.0, abs=0.01)
    assert rhat(np.vstack([rng.normal(size=1000), rng.normal(size=1000) + 3])) > 1.5  # chains that disagree
    low, high = hpd(rng.normal(size=200000), 0.9)
    assert low == pytest.approx(-1.645, abs=0.02) and high == pytest.approx(1.645, abs=0.02)
    low, high = hpd(rng.exponential(size=200000), 0.9)  # skewed: the HPD starts at the mode, zero
    assert low == pytest.approx(0.0, abs=0.01) and high == pytest.approx(-np.log(0.1), abs=0.03)


def test_results_views_and_bands(ar1_posterior):
    post = ar1_posterior
    table = post.summary()
    assert "R-hat" in table.columns and "Log marginal data density" in " ".join(table.notes)
    assert post.to_frame().shape == (6000, 4) and isinstance(post, Posterior)
    irf = post.irf(8, draws=50)
    assert np.all(irf.lower <= irf.values + 1e-12) and np.all(irf.values <= irf.upper + 1e-12)
    assert irf.values[0, 0, 0] == pytest.approx(0.5)  # a one-standard-deviation shock, the size fixed at 0.5
    assert irf.values[1, 0, 0] == pytest.approx(0.5 * np.median(post["rho"]), rel=0.05)
    assert len(post.plot().panels) == 1 and len(post.trace().panels[0].layers) == 2

    from pymetrix.ui.render import view_sections

    titles = [t for t, _ in view_sections(post)]
    assert titles[0] == "Summary" and "Draws" in titles
    if importlib.util.find_spec("matplotlib") is not None:
        assert "Graph" in titles and "Chains" in titles


def test_identification_against_the_analytic_jacobian(ar1_fit):
    model, data, _ = ar1_fit
    fit = model.estimate(data, ["x"], estimate=["rho", "sd_e"])
    idn = fit.identification(lags=3)
    rho, s = idn.at["rho"], idn.at["sd_e"]
    g0 = s**2 / (1 - rho**2)
    d_g0 = np.array([2 * rho * s**2 / (1 - rho**2) ** 2, 2 * s / (1 - rho**2)])  # d gamma0 / d (rho, s)
    expected = [np.zeros(2), d_g0]  # the mean (zero), the variance, then the autocovariances rho^j gamma0
    for j in range(1, 4):
        expected.append(np.array([j * rho ** (j - 1) * g0, 0.0]) + rho**j * d_g0)
    assert np.allclose(idn.jacobian, np.vstack(expected), rtol=1e-5, atol=1e-8)
    assert idn.identified and idn.problems == ()

    product = Model([Equation("x ~ a*b*x[-1] + e", params=["a", "b"])], params={"a": 0.8, "b": 0.9}, roles={"e": Role.SHOCK})
    bad = product.estimate(data, ["x"], estimate=["a", "b", "sd_e"]).identification()
    assert bad.rank == 2 and not bad.identified and set(bad.problems) == {"a", "b"}
    assert bad.closest[0] == "b" and "not identified" in str(bad.summary())


def test_errors(ar1_fit):
    model, data, fit = ar1_fit
    with pytest.raises(ModelError, match="give every estimated parameter a prior"):
        model.estimate(data, ["x"], estimate=["rho", "sd_e"], priors={"rho": Normal(0.5, 0.2)}).mcmc(draws=20)
    walk = Model([Equation("x ~ x[-1] + a*e", params=["a"])], params={"a": 1.0}, roles={"e": Role.SHOCK})
    with pytest.raises(ModelError, match="unit root"):
        walk.estimate(data, ["x"], estimate=["a"]).identification()
    with pytest.raises(ValueError):
        fit.mcmc(draws=5)


def test_the_dialogs():
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import CheckIdentification, EstimateModel, RunMCMC

    nk = templates.new_keynesian()
    truth = nk.solve().simulate(120, nk.shock_sd, seed=11)
    truth.index = _quarters(120)
    s = Session()
    s.objects["nk"] = nk
    s.objects["data"] = nk.measurement.apply(Dataset.from_frame(truth))
    s.run(EstimateModel("nk", estimate="kappa", priors="kappa=Gamma(0.3, 0.1)"))
    assert s["estimated"].priors == {"kappa": Gamma(0.3, 0.1)}
    s.run(RunMCMC("estimated", draws=40, chains=1, seed=2))
    s.run(CheckIdentification("estimated", lags=2))
    script = s.script()
    assert "estimated = nk.estimate(data, estimate=['kappa'], priors={'kappa': priors.Gamma(mean=0.3, sd=0.1)})" in script
    assert "posterior = estimated.mcmc(draws=40, chains=1, seed=2)" in script
    assert "identification = estimated.identification(lags=2)" in script
    assert s["posterior"].draws.shape == (1, 40, 1) and s["identification"].identified
    with pytest.raises(ValueError, match="give priors as"):
        EstimateModel("nk", priors="kappa=Weibull(1, 2)").run(s)


def test_the_proposal_takes_the_shape_of_the_burn_in_draws():
    from pymetrix.models.bayes import _adapted

    rng = np.random.default_rng(0)
    ridge = rng.multivariate_normal([0, 0], [[1.0, 0.95], [0.95, 1.0]], size=2000)  # a long narrow ridge
    old = np.eye(2)
    factor, c = _adapted(ridge, old, 1.19)
    cov = factor @ factor.T
    assert cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]) > 0.8 and c == 1.19  # along the ridge, then tuned again
    stuck = np.zeros((2000, 2))  # a chain that never moved says nothing: keep the old proposal
    assert _adapted(stuck, old, 1.19)[0] is old


def test_the_summary_says_when_the_chains_disagree(ar1_fit, ar1_posterior):
    from dataclasses import replace

    assert not any("disagree" in n for n in ar1_posterior.summary().notes)
    draws = ar1_posterior.draws.copy()
    draws[1] += 0.2  # the second chain somewhere else
    apart = replace(ar1_posterior, draws=draws)
    assert any(n.startswith("The chains disagree on rho") for n in apart.summary().notes)
