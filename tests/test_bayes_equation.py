"""Bayesian single equations, checked against exact posteriors: flat priors (least squares, and the multivariate t's
variance); normal priors with an inverse gamma on the error (the variance integrated by quadrature, the coefficients
in closed form given it); a Beta prior, a logit and a Poisson model (two-parameter posteriors on a grid); the Jacobians
of the sampled log sigma; chains' agreement; refusals."""

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, special, stats

from pymetrix import Dataset, Equation, Logit, Poisson
from pymetrix.estimation import Bayes
from pymetrix.estimation.priors import Beta, InvGamma, InvGamma1, Normal

T = 60


def _data(seed=0, kind="gaussian"):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=T)
    if kind == "gaussian":
        y = 1.0 + 0.6 * x + rng.normal(scale=0.8, size=T)
    elif kind == "logit":
        y = (rng.uniform(size=T) < special.expit(-0.3 + 1.2 * x)).astype(float)
    else:
        y = rng.poisson(np.exp(0.4 + 0.5 * x)).astype(float)
    return Dataset.from_frame(pd.DataFrame({"y": y, "x": x}, index=pd.period_range("1960Q1", periods=T, freq="Q"))), x, y


def _mc(draws, value, se_mult=4):
    """within se_mult Monte Carlo standard errors (draws' s.d. over the root of their effective size)"""
    return abs(draws.mean() - value) < se_mult * draws.std() / np.sqrt(len(draws) / 10)


def test_flat_priors_give_least_squares_and_the_t_variance():
    data, x, y = _data()
    fit = Equation("y ~ x").fit(data, method=Bayes(draws=20_000, chains=2, burn=200))
    ols = Equation("y ~ x").fit(data)
    X = np.column_stack([np.ones(T), x])
    s2 = np.sum(ols.resid.to_numpy() ** 2) / (T - 2)
    var = s2 * np.linalg.inv(X.T @ X) * (T - 2) / (T - 4)  # the posterior is t with T - k degrees of freedom
    assert np.all(np.abs(fit.params.values - ols.params.values) < 4 * np.sqrt(np.diag(var) / 40_000))  # Monte Carlo error
    np.testing.assert_allclose(np.diag(fit.params.cov), np.diag(var), rtol=0.04)
    assert fit.extra["sampler"] == "Gibbs" and "Gibbs" in str(fit.summary())
    post = fit.posterior()
    assert (post["R-hat"] < 1.01).all() and (post["ESS"] > 10_000).all()


def _exact_conjugate(x, y, priors, a, b):
    """Posterior means of (const, x) and sigma^2 with normal priors and sigma^2 ~ IG(a, b): quadrature over sigma^2."""
    X = np.column_stack([np.ones(T), x])
    m0, V0 = np.array([p.mean for p in priors]), np.diag([p.sd**2 for p in priors])
    grid = np.exp(np.linspace(np.log(0.05), np.log(5), 3000))
    lp = np.array([stats.invgamma.logpdf(s, a, scale=b) + stats.multivariate_normal.logpdf(y, X @ m0, s * np.eye(T) + X @ V0 @ X.T)
                   for s in grid])
    wt = np.exp(lp - lp.max())
    cond = np.array([np.linalg.solve(np.linalg.inv(V0) + X.T @ X / s, np.linalg.solve(V0, m0) + X.T @ y / s) for s in grid])
    norm = integrate.trapezoid(wt, grid)
    return integrate.trapezoid(wt[:, None] * cond, grid, axis=0) / norm, integrate.trapezoid(wt * grid, grid) / norm


@pytest.mark.parametrize("sampler, error", [("auto", "sigma"), ("auto", "sigma2"), ("rwmh", "sigma"), ("rwmh", "sigma2")])
def test_normal_priors_against_the_exact_posterior(sampler, error):
    data, x, y = _data(1)
    priors = [Normal(0.0, 1.0), Normal(0.5, 0.2)]
    if error == "sigma":
        p_err = InvGamma1(0.8, 0.5)
        nu, s = p_err.nu_s
        a, b = nu / 2, s / 2
    else:
        p_err = InvGamma(0.6, 0.4)
        a = (0.6 / 0.4) ** 2 + 2
        b = 0.6 * (a - 1)
    exact_b, exact_s2 = _exact_conjugate(x, y, priors, a, b)
    fit = Equation("y ~ x").fit(data, method=Bayes(prior={"const": priors[0], "x": priors[1], error: p_err}, draws=20_000,
                                                    chains=2, sampler=sampler, seed=3))
    assert fit.extra["sampler"] == ("Gibbs" if sampler == "auto" else "random-walk Metropolis-Hastings")
    for j, name in enumerate(["const", "x"]):
        assert _mc(fit.draws[name], exact_b[j]), (name, fit.draws[name].mean(), exact_b[j])
    assert _mc(fit.draws["sigma"] ** 2, exact_s2)  # the Jacobian of the sampled log sigma is right
    if sampler == "rwmh":
        assert 0.15 < fit.stats["acceptance"] < 0.4


def _grid_means(logpost, a_range, b_range, n=400):
    A, B = np.meshgrid(np.linspace(*a_range, n), np.linspace(*b_range, n), indexing="ij")
    lp = logpost(A, B)
    w = np.exp(lp - lp.max())
    return (w * A).sum() / w.sum(), (w * B).sum() / w.sum(), np.sqrt((w * (A - (w * A).sum() / w.sum()) ** 2).sum() / w.sum())


def test_a_beta_prior_by_metropolis_hastings():
    data, x, y = _data(2)
    fit = Equation("y ~ x - 1").fit(data, method=Bayes(prior={"x": Beta(0.5, 0.2), "sigma": InvGamma1(1.0, 1.0)},
                                                       draws=20_000, chains=2, seed=1))
    assert fit.extra["sampler"] == "random-walk Metropolis-Hastings"  # a Beta prior has no exact Gibbs step
    nu, s = InvGamma1(1.0, 1.0).nu_s

    def logpost(b, sig):
        e = y[None, None, :] - b[..., None] * x[None, None, :]
        return (stats.beta.logpdf(b, *_beta_ab(0.5, 0.2)) - (nu + 1) * np.log(sig) - s / (2 * sig**2)
                - T * np.log(sig) - 0.5 * np.sum(e**2, axis=-1) / sig**2)

    mb, ms, sd_b = _grid_means(logpost, (0.001, 0.999), (0.4, 2.5))
    assert abs(fit.draws["x"].mean() - mb) < 0.1 * sd_b and abs(fit.draws["sigma"].mean() - ms) < 0.01
    assert (fit.draws["x"] > 0).all() and (fit.draws["x"] < 1).all()


def _beta_ab(mean, sd):
    common = mean * (1 - mean) / sd**2 - 1
    return mean * common, (1 - mean) * common


@pytest.mark.parametrize("kind", ["logit", "poisson"])
def test_logit_and_poisson_against_a_grid(kind):
    data, x, y = _data(3, kind)
    family = Logit() if kind == "logit" else Poisson()
    fit = Equation("y ~ x", family=family).fit(data, method=Bayes(prior={"const": Normal(0, 2), "x": Normal(0, 2)},
                                                                    draws=20_000, chains=2, seed=2))

    def logpost(a, b):
        eta = a[..., None] + b[..., None] * x[None, None, :]
        ll = (y * eta - np.logaddexp(0, eta)) if kind == "logit" else (y * eta - np.exp(eta))
        return stats.norm.logpdf(a, 0, 2) + stats.norm.logpdf(b, 0, 2) + ll.sum(axis=-1)

    ml = Equation("y ~ x", family=family).fit(data, method=__import__("pymetrix").ML())
    c, se = ml.params.values, np.sqrt(np.diag(ml.params.cov))
    ma, mb, sd_a = _grid_means(logpost, (c[0] - 6 * se[0], c[0] + 6 * se[0]), (c[1] - 6 * se[1], c[1] + 6 * se[1]))
    assert abs(fit.params.values[0] - ma) < 0.08 * sd_a and abs(fit.params.values[1] - mb) < 0.08 * se[1]
    assert (fit.posterior()["R-hat"] < 1.02).all()


def test_refusals():
    data, _, _ = _data()
    with pytest.raises(ValueError, match="sampler is"):
        Equation("y ~ x").fit(data, method=Bayes(sampler="slice"))
    with pytest.raises(ValueError, match="not coefficients"):
        Equation("y ~ x").fit(data, method=Bayes(prior={"z": Normal(0, 1)}))
    with pytest.raises(ValueError, match="sigma or on sigma2"):
        Equation("y ~ x").fit(data, method=Bayes(prior={"sigma": InvGamma1(1, 1), "sigma2": InvGamma(1, 1)}))


def test_the_equation_dialog_estimates_by_bayes():
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import FitEquation
    from pymetrix.estimation.priors import parse
    from pymetrix.forms import settings_form
    from pymetrix.plugins import registry

    entry = registry("methods").get("Bayes")
    assert entry.kind == "bayes" and [s.name for s in settings_form(entry)] == ["prior", "draws", "sampler", "seed", "chains", "burn", "metric"]
    assert "nuts" in next(s for s in settings_form(entry) if s.name == "sampler").choices
    rng = np.random.default_rng(4)
    n = 200
    x = rng.normal(size=n)
    y = 1 + 0.5 * x + rng.normal(size=n)
    session = Session()
    session.objects["data"] = Dataset.from_frame(pd.DataFrame({"y": y, "x": x, "b": (y > 1).astype(float)},
                                                              index=pd.period_range("2000Q1", periods=n, freq="Q", name="time")))
    typed = "x=Normal(0.5, 0.1); sigma=InvGamma1(1, 1)"
    fit = session.run(FitEquation("y ~ x", method="Bayes", method_settings=f"prior='{typed}', draws=1500, seed=3"))
    same = Equation("y ~ x").fit(session["data"], method=Bayes(prior=parse(typed), draws=1500, seed=3))
    assert np.allclose(fit.params.values, same.params.values)  # the dialog's fit is the code's
    assert "method=Bayes(prior={'x': priors.Normal(mean=0.5, sd=0.1), 'sigma': priors.InvGamma1(mean=1.0, sd=1.0)}, draws=1500, seed=3)" \
        in session.script()
    namespace: dict = {"data": session["data"]}
    exec(session.script(), namespace)  # the recorded script replays without the typed text
    assert np.allclose(namespace["fit"].params.values, fit.params.values)
    logit = session.run(FitEquation("b ~ x", method="Bayes", family="logit", method_settings="draws=800", name="lfit"))
    assert type(logit.equation.family).__name__ == "Logit"
    for cmd, message in ((FitEquation("y ~ x", method="Bayes", family="tobit"), "not tobit"),
                         (FitEquation("y ~ x", method="Bayes", ar="1"), "ARMA errors or GARCH"),
                         (FitEquation("y ~ x", method="Bayes", cov="white"), "posterior's uncertainty"),
                         (FitEquation("y ~ x", method="Bayes", method_settings="prior='x=Normal(1)'"), "two numbers")):
        problems = cmd.validate(session)
        assert problems and message in problems[0], problems
    assert Bayes(prior="").prior is None and Bayes(prior="x=Normal(0, 1)").prior == {"x": parse("x=Normal(0, 1)")["x"]}
