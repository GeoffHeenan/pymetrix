"""The No-U-Turn sampler (pymetrix.nuts), checked independently: a correlated, ill-scaled Gaussian's mean and covariance;
the conjugate normal-inverse-gamma regression's closed-form posterior; Bayesian single equations (flat priors against
least squares and the t variance, a logit against random-walk Metropolis-Hastings, the analytic gradients against
finite differences); a linearised model's posterior against the exact one on a grid; Stan's warmup windows; the
transformations off the bounds; the dialogs and refusals."""

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, special, stats

from pymetrix import Dataset, Equation, Logit, Model, Poisson, Probit
from pymetrix.errors import SpecificationError
from pymetrix.estimation import Bayes
from pymetrix.estimation.priors import Beta, Gamma, InvGamma, InvGamma1, Normal
from pymetrix.metadata import Role
from pymetrix.models.bayes import ess, rhat
from pymetrix.nuts import Transform, sample, windows


def _within(draws, value, sd, n_eff, k=4.0):
    return abs(np.mean(draws) - value) < k * sd / np.sqrt(n_eff)


def test_a_correlated_ill_scaled_gaussian():
    sd = np.array([0.1, 1.0, 10.0])
    R = np.array([[1.0, 0.9, 0.5], [0.9, 1.0, 0.3], [0.5, 0.3, 1.0]])
    C = np.outer(sd, sd) * R
    P = np.linalg.inv(C)
    mu = np.array([1.0, -2.0, 50.0])

    def f(x):
        d = x - mu
        g = -P @ d
        return 0.5 * float(d @ g), g

    run = sample(f, np.tile(mu + sd, (3, 1)), draws=400, warmup=300, seed=3)  # from a unit metric: warmup finds the scales
    x = run.draws.reshape(-1, 3)
    for j in range(3):
        n_eff = ess(run.draws[:, :, j])
        assert n_eff > 200 and rhat(run.draws[:, :, j]) < 1.02
        assert _within(x[:, j], mu[j], sd[j], n_eff)
    # the covariance, as correlations and variances, each within four of its Monte Carlo standard errors at the
    # smallest effective sample size (a correlation's about (1 - rho^2) / sqrt(n), a variance's relative sqrt(2 / n))
    n = min(ess(run.draws[:, :, j]) for j in range(3))
    assert np.all(np.abs(np.corrcoef(x.T) - R) <= 4 * (1 - R**2) / np.sqrt(n) + 1e-12)
    assert np.allclose(np.var(x, axis=0, ddof=1) / sd**2, 1, atol=4 * np.sqrt(2 / n))
    assert run.divergent.sum() == 0 and np.allclose(np.sqrt(run.inv_metric.mean(axis=0)) / sd, 1, atol=0.35)
    info = run.info()
    assert info["max depth"] == 10 and len(info["step size"]) == 3 and 0.6 < np.mean(info["mean acceptance statistic"]) < 0.99


def test_the_conjugate_regression_posterior():
    """beta | s2 ~ N(b0, s2 V0), s2 ~ IG(a0, c0): beta's marginal posterior is a multivariate t, s2's an inverse gamma."""
    rng = np.random.default_rng(4)
    n = 50
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = X @ [1.0, 0.5] + rng.normal(scale=0.7, size=n)
    b0, V0, a0, c0 = np.array([0.0, 0.0]), np.diag([4.0, 1.0]), 3.0, 2.0
    V0i = np.linalg.inv(V0)
    Vn = np.linalg.inv(V0i + X.T @ X)
    bn = Vn @ (V0i @ b0 + X.T @ y)
    an, cn = a0 + n / 2, c0 + 0.5 * (y @ y + b0 @ V0i @ b0 - bn @ np.linalg.inv(Vn) @ bn)

    def f(t):  # (beta, log s2), the Jacobian s2 included; s2^(-k/2) from beta's prior (k = 2)
        beta, ls2 = t[:2], t[2]
        s2 = np.exp(ls2)
        e, d = y - X @ beta, beta - b0
        q = e @ e + d @ V0i @ d
        lp = -(n / 2 + 1 + 1 + a0) * ls2 - (c0 + 0.5 * q) / s2 + ls2
        g = np.append((X.T @ e - V0i @ d) / s2, -(n / 2 + 1 + 1 + a0) + (c0 + 0.5 * q) / s2 + 1)
        return lp, g

    run = sample(f, np.zeros((2, 3)), draws=1500, warmup=500, seed=1)
    beta = run.draws[:, :, :2].reshape(-1, 2)
    s2 = np.exp(run.draws[:, :, 2].ravel())
    cov = cn / an * Vn * (2 * an) / (2 * an - 2)  # the t's covariance
    for j in range(2):
        assert _within(beta[:, j], bn[j], np.sqrt(cov[j, j]), ess(run.draws[:, :, j]))
    scale = np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    assert np.allclose((np.cov(beta.T) - cov) / scale, 0, atol=0.1)
    mean_s2, sd_s2 = cn / (an - 1), cn / ((an - 1) * np.sqrt(an - 2))
    assert _within(s2, mean_s2, sd_s2, ess(run.draws[:, :, 2])) and s2.std() == pytest.approx(sd_s2, rel=0.12)


def _data(seed, kind="gaussian", T=60):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=T)
    if kind == "gaussian":
        y = 1.0 + 0.6 * x + rng.normal(scale=0.8, size=T)
    elif kind in ("logit", "probit"):
        y = (rng.uniform(size=T) < special.expit(-0.3 + 1.2 * x)).astype(float)
    else:
        y = rng.poisson(np.exp(0.4 + 0.5 * x)).astype(float)
    return Dataset.from_frame(pd.DataFrame({"y": y, "x": x}, index=pd.period_range("1960Q1", periods=T, freq="Q"))), x, y


def test_flat_priors_give_least_squares_and_the_t_variance():
    data, x, y = _data(0)
    fit = Equation("y ~ x").fit(data, method=Bayes(sampler="nuts", draws=1000, chains=2, burn=300, seed=1))
    ols = Equation("y ~ x").fit(data)
    T, k = len(y), 2
    s2 = float(np.sum(ols.resid.to_numpy() ** 2) / (T - k))
    sd = np.sqrt(np.diag(ols.params.cov) * (T - k) / (T - k - 2))
    post = fit.posterior()
    for j, name in enumerate(["const", "x"]):
        assert _within(fit.draws[name], ols.params.values[j], sd[j], post.loc[name, "ESS"])
        assert fit.draws[name].std() == pytest.approx(sd[j], rel=0.1)
    sigma2 = fit.draws["sigma"].to_numpy() ** 2  # 1/sigma^2: sigma^2 | y is an inverse gamma ((T-k)/2, (T-k) s2/2)
    assert _within(sigma2, (T - k) * s2 / (T - k - 2), s2 * np.sqrt(2 / (T - k - 4)), post.loc["sigma", "ESS"])
    assert fit.extra["sampler"] == "No-U-Turn (NUTS)" and "divergences" in fit.stats and "nuts" in fit.extra
    assert "No-U-Turn sampler" in str(fit.summary())


def test_a_logit_agrees_with_metropolis_hastings():
    data, _, _ = _data(3, "logit")
    prior = {"const": Normal(0, 2), "x": Gamma(1, 1)}  # a bounded prior: sampled through its log
    nuts = Equation("y ~ x", family=Logit()).fit(data, method=Bayes(prior=prior, sampler="nuts", draws=1000, chains=2, burn=400, seed=2))
    mh = Equation("y ~ x", family=Logit()).fit(data, method=Bayes(prior=prior, sampler="rwmh", draws=8000, chains=2, seed=2))
    a, b = nuts.posterior(), mh.posterior()
    for name in ("const", "x"):
        se = np.sqrt(a.loc[name, "s.d."] ** 2 / a.loc[name, "ESS"] + b.loc[name, "s.d."] ** 2 / b.loc[name, "ESS"])
        assert abs(a.loc[name, "mean"] - b.loc[name, "mean"]) < 4 * se
        assert a.loc[name, "s.d."] == pytest.approx(b.loc[name, "s.d."], rel=0.15)
    assert (nuts.draws["x"] > 0).all() and (a["R-hat"] < 1.02).all()


@pytest.mark.parametrize("kind", ["gaussian", "logit", "probit", "poisson"])
def test_the_gradients_are_the_log_posteriors(kind):
    from pymetrix.estimation.bayes_equation import _gradient_posterior

    _, x, y = _data(5, kind, T=40)
    X = np.column_stack([np.ones(len(y)), x])
    w = np.random.default_rng(0).uniform(0.5, 1.5, len(y))
    family = {"logit": Logit(), "probit": Probit(), "poisson": Poisson()}.get(kind)
    gaussian = family is None
    priors = [Normal(0.5, 1.0), Beta(0.5, 0.2)]
    for key, error in ((("sigma", InvGamma1(0.8, 0.5)), ("sigma2", InvGamma(0.6, 0.4)), ("sigma", None)) if gaussian else (("sigma", None),)):
        f = _gradient_posterior(family, gaussian, y, X, w, priors, key, error)
        theta = np.array([0.4, 0.3] + ([np.log(0.9)] if gaussian else []))
        _, g = f(theta)
        h = 1e-6
        numeric = [(f(theta + h * e)[0] - f(theta - h * e)[0]) / (2 * h) for e in np.eye(len(theta))]
        assert np.allclose(g, numeric, rtol=1e-6, atol=1e-6)


def test_transformations_off_the_bounds():
    tr = Transform.of([(0, 1), (0, np.inf), (-np.inf, 2), (-np.inf, np.inf), (-3, 5)])
    z = np.array([0.3, -0.2, 0.5, 1.0, -1.2])
    assert np.allclose(tr.to_z(tr.to_x(z)), z)
    f = tr.wrap(lambda v: (float(-0.5 * v @ v), -v))
    lp, g = f(z)
    assert lp == pytest.approx(-0.5 * tr.to_x(z) @ tr.to_x(z) + tr.log_jacobian(z))
    numeric = [(f(z + 1e-6 * e)[0] - f(z - 1e-6 * e)[0]) / 2e-6 for e in np.eye(5)]
    assert np.allclose(g, numeric, atol=1e-6)


def test_stans_warmup_windows():
    assert windows(1000) == (75, [100, 150, 250, 450, 950])
    assert windows(100) == (15, [90])
    assert windows(10) == (10, [])


def _ar1(n=120):
    rng = np.random.default_rng(1)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + 0.5 * rng.normal()
    data = Dataset.from_frame(pd.DataFrame({"x": x}, index=pd.period_range("1950Q1", periods=n, freq="Q", name="time")))
    model = Model([Equation("x ~ rho*x[-1] + e", params=["rho"])], params={"rho": 0.3}, roles={"e": Role.SHOCK})
    return model.estimate(data, ["x"], estimate=["rho"], shock_sd={"e": 0.5}, priors={"rho": Beta(0.5, 0.2)})


@pytest.fixture(scope="module")
def ar1():
    fit = _ar1()
    return fit, fit.mcmc(draws=150, chains=2, burn=100, seed=1, sampler="nuts")


def test_a_linearised_models_posterior_against_the_exact_one(ar1):
    fit, post = ar1
    g = np.linspace(0.4, 0.95, 551)  # the posterior is far inside
    lp = np.array([fit.log_posterior([v]) for v in g])
    top = lp[np.isfinite(lp)].max()
    w = np.exp(np.where(np.isfinite(lp), lp - top, -np.inf))
    z = integrate.trapezoid(w, g)
    mean = integrate.trapezoid(w * g, g) / z
    sd = np.sqrt(integrate.trapezoid(w * (g - mean) ** 2, g) / z)
    assert _within(post["rho"], mean, sd, ess(post.draws[:, :, 0]))
    assert post["rho"].std() == pytest.approx(sd, rel=0.2)
    assert np.allclose(post.logpost.ravel(), [fit.log_posterior([v]) for v in post["rho"]])  # in the parameters' own terms


def test_the_posterior_records_the_sampler(ar1):
    fit, post = ar1
    assert post.sampler == "nuts" and post.draws.shape == (2, 150, 1)
    method = post.provenance.method
    assert method["sampler"].startswith("No-U-Turn") and method["warmup"] == 100 and len(method["divergences"]) == 2
    assert "No-U-Turn sampler" in str(post.summary()) and post.trace().title == "NUTS chains"
    with pytest.raises(SpecificationError, match="sampler"):
        fit.mcmc(draws=20, sampler="slice")
    with pytest.raises(SpecificationError, match="target_accept"):
        fit.mcmc(draws=20, sampler="nuts", target_accept=1.0)


def test_the_dialogs(ar1):
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import RunMCMC
    from pymetrix.forms import settings_form
    from pymetrix.plugins import registry

    fit, _ = ar1
    s = Session()
    s.objects["estimated"] = fit
    s.run(RunMCMC("estimated", draws=20, chains=1, seed=3, burn=20, sampler="nuts", target_accept=0.9, name="post"))
    assert s["post"].sampler == "nuts"
    assert "post = estimated.mcmc(draws=20, chains=1, burn=20, seed=3, sampler='nuts', target_accept=0.9)" in s.script()
    entry = registry("methods").get("Bayes")
    assert "nuts" in next(f for f in settings_form(entry) if f.name == "sampler").choices


# --- the dense metric ---------------------------------------------------------------------------------------------

def _correlated(k=6, r=0.99):
    sd = np.logspace(-1, 1, k)
    R = np.full((k, k), r)
    np.fill_diagonal(R, 1.0)
    C = np.outer(sd, sd) * R
    P = np.linalg.inv(C)
    mu = np.linspace(-1, 1, k)

    def f(x):
        g = -P @ (x - mu)
        return 0.5 * float((x - mu) @ g), g

    return f, mu, sd, C


def _whitened(M, C):
    """The eigenvalues of ``M`` in the coordinates where ``C`` is the identity: all 1 when ``M`` is ``C``; their spread
    is how far ``M`` is from ``C`` in its worst direction, which an elementwise ratio misses when ``C`` is nearly rank one."""
    Li = np.linalg.inv(np.linalg.cholesky(C))
    return np.linalg.eigvalsh(Li @ M @ Li.T)


def test_a_dense_metric_undoes_a_correlation_of_099(monkeypatch):
    """Six coordinates, every pair correlated 0.99, scales from 0.1 to 10 (condition number about 1e6): the dense metric
    adapts to the covariance and needs a few leapfrog steps a draw; the diagonal one cannot undo the correlation.

    Covariances are compared whitened by ``C`` (``_whitened``), not elementwise: with every pair at 0.99 an elementwise
    ratio checks one number, the variance along ``1``, and fails a run whose estimate is merely noisy. The bounds are
    statistical, so they hold whatever BLAS kernel rounds the arithmetic (and so whichever draws the seed gives).
    The adapted metric is also checked exactly: Stan's regularised covariance of the draws of each chain's last slow
    window, recorded as the sampler hands them over, which no tolerance can miss (a dropped n/(n+5), say)."""
    from pymetrix import nuts

    windows_seen, regularised = [], nuts._regularised
    monkeypatch.setattr(nuts, "_regularised", lambda w, dense, current: windows_seen.append(np.array(w)) or regularised(w, dense, current))
    f, mu, sd, C = _correlated()
    runs = {m: sample(f, np.tile(mu + sd, (c, 1)), draws=d, warmup=300, seed=4, metric=m)
            for m, c, d in (("diag", 1, 300), ("dense", 2, 1000))}
    per_gradient = {}
    for m, run in runs.items():
        n_eff = np.array([ess(run.draws[:, :, j]) for j in range(6)])
        per_gradient[m] = n_eff.min() / run.leapfrog.sum()  # leapfrog counts the kept draws' steps only
        if m == "dense":
            x = run.draws.reshape(-1, 6)
            assert all(_within(x[:, j], mu[j], sd[j], n_eff[j]) for j in range(6))
            # The draws' covariance: its whitened eigenvalues lie within the Marchenko-Pastur edges (1 +- sqrt(k/n))^2
            # of a sample of n = ESS, widened 3 times (as _within's k); about [0.65, 1.43] here, so C/2 or 2C fails.
            # Over 1000 seeds the extremes were 0.81 and 1.22.
            r = 3 * np.sqrt(6 / n_eff.min())
            e = _whitened(np.cov(x.T), C)
            assert (1 - r) ** 2 < e.min() and e.max() < (1 + r) ** 2, (e, r)
            # The adapted covariance, from the last slow window's 100 autocorrelated warmup draws: within a factor of 4
            # of C in every direction (over 2000 chains, 0.34 to 2.53), where the identity is off by 8400 and diag(C)
            # by 100: the correlation is undone. Its scale, the mean of those eigenvalues (an average over six
            # directions, so far less noisy: sd 0.085 over those chains, 0.79 to 1.31), within 30% of C's.
            assert run.inv_metric.shape == (2, 6, 6)
            _, ends = windows(300)  # each chain's windows in turn: the last of each, the dense run's two chains last
            last = windows_seen[-2 * len(ends):][len(ends) - 1::len(ends)]
            assert len(last) == 2
            for M, X in zip(run.inv_metric, last):
                n = len(X)
                assert n == ends[-1] - ends[-2]
                assert np.allclose(M, n / (n + 5) * np.cov(X.T) + 1e-3 * 5 / (n + 5) * np.eye(6), rtol=1e-10, atol=0)
                e = _whitened(M, C)
                assert 0.25 < e.min() and e.max() < 4, e
                assert 0.7 < e.mean() < 1 / 0.7, e
            assert run.info()["metric"] == "dense" and "dense metric" in run.notes()[0] and n_eff.min() > x.shape[0] / 2
    assert per_gradient["dense"] > 10 * per_gradient["diag"], per_gradient
    assert runs["diag"].inv_metric.shape == (1, 6) and runs["diag"].info()["metric"] == "diagonal"


def test_a_fixed_dense_metric_and_the_refusals():
    f, mu, sd, C = _correlated(k=3)
    run = sample(f, mu[None, :], draws=300, warmup=60, seed=1, metric="dense", inv_metric=C, adapt_metric=False)
    assert np.allclose(run.inv_metric[0], C)  # kept: only the step size adapted
    assert run.depth.mean() < 3 and run.divergent.sum() == 0  # a standard normal in the metric's coordinates
    x = run.draws.reshape(-1, 3)
    assert all(_within(x[:, j], mu[j], sd[j], ess(run.draws[:, :, j])) for j in range(3))
    with pytest.raises(SpecificationError, match="metric is"):
        sample(f, mu[None, :], draws=5, warmup=5, metric="full")
    with pytest.raises(SpecificationError, match="metric='dense'"):
        sample(f, mu[None, :], draws=5, warmup=5, inv_metric=C)
    with pytest.raises(SpecificationError, match="positive definite"):
        sample(f, mu[None, :], draws=5, warmup=5, metric="dense", inv_metric=-C)
    with pytest.raises(SpecificationError, match="3 x 3"):
        sample(f, mu[None, :], draws=5, warmup=5, metric="dense", inv_metric=np.eye(2))


def test_a_dense_metric_for_a_bayesian_single_equation():
    """A regressor far from zero: the intercept and slope correlate about -0.99. Flat priors: the posterior is the t
    around least squares; the dense metric and the mode's inverse Hessian take far fewer gradients a draw."""
    rng = np.random.default_rng(7)
    T = 60
    x = 20 + rng.normal(size=T)
    y = 1.0 + 0.6 * x + rng.normal(scale=0.8, size=T)
    data = Dataset.from_frame(pd.DataFrame({"y": y, "x": x}, index=pd.period_range("1960Q1", periods=T, freq="Q")))
    ols = Equation("y ~ x").fit(data)
    sd = np.sqrt(np.diag(ols.params.cov) * (T - 2) / (T - 4))
    steps = {}
    for metric in ("diag", "dense", "mode"):
        fit = Equation("y ~ x").fit(data, method=Bayes(sampler="nuts", metric=metric, draws=300, chains=2, burn=200, seed=3))
        steps[metric] = fit.extra["nuts"]["gradient evaluations"]
        if metric != "diag":
            post = fit.posterior()
            for j, name in enumerate(["const", "x"]):
                assert _within(fit.draws[name], ols.params.values[j], sd[j], post.loc[name, "ESS"])
                assert fit.draws[name].std() == pytest.approx(sd[j], rel=0.15)
    assert steps["dense"] < steps["diag"] / 3 and steps["mode"] < steps["diag"] / 3, steps
    with pytest.raises(SpecificationError, match="metric"):
        Equation("y ~ x").fit(data, method=Bayes(sampler="nuts", metric="full", draws=100))


def test_a_linearised_models_posterior_with_the_dense_and_mode_metrics():
    fit = _ar1()
    g = np.linspace(0.4, 0.95, 551)
    lp = np.array([fit.log_posterior([v]) for v in g])
    w = np.exp(lp - lp[np.isfinite(lp)].max())
    z = integrate.trapezoid(w, g)
    mean = integrate.trapezoid(w * g, g) / z
    sd = np.sqrt(integrate.trapezoid(w * (g - mean) ** 2, g) / z)
    for metric, burn in (("dense", 60), ("mode", 20)):
        post = fit.mcmc(draws=120, chains=2, burn=burn, seed=1, sampler="nuts", metric=metric)
        assert _within(post["rho"], mean, sd, ess(post.draws[:, :, 0]))
        assert post.provenance.method["metric"].startswith("dense")
    assert "inverse Hessian" in post.provenance.method["metric"]
    with pytest.raises(SpecificationError, match="metric"):
        fit.mcmc(draws=20, sampler="nuts", metric="full")


def test_the_mcmc_dialog_takes_the_metric(ar1):
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import RunMCMC
    from pymetrix.forms import settings_form
    from pymetrix.plugins import registry

    fit, _ = ar1
    s = Session()
    s.objects["estimated"] = fit
    command = RunMCMC("estimated", draws=20, chains=1, seed=3, burn=20, sampler="nuts", metric="mode", name="post")
    s.run(command)
    line = "post = estimated.mcmc(draws=20, chains=1, burn=20, seed=3, sampler='nuts', metric='mode')"
    assert line in s.script()
    direct = fit.mcmc(draws=20, chains=1, burn=20, seed=3, sampler="nuts", metric="mode")
    assert np.array_equal(s["post"].draws, direct.draws)  # the dialog's result is the API call's
    replay = {"estimated": fit}
    exec(line, {}, replay)  # the generated script replays
    assert np.array_equal(replay["post"].draws, direct.draws)
    entry = registry("methods").get("Bayes")
    assert next(f for f in settings_form(entry) if f.name == "metric").choices == ("diag", "dense", "mode")
