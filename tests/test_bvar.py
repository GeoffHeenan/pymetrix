"""Bayesian VARs: the conjugate Minnesota prior, checked against exact results and least squares."""

from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from pymetrix import VAR, Bayes, Dataset, FittedBVAR, Minnesota
from pymetrix.bvar import _niw

DATA = Path(__file__).parents[1] / "examples" / "sample_macro.parquet"


@pytest.fixture(scope="module")
def data():
    return Dataset.read(str(DATA)).with_series(ly="100*log(y)", lc="100*log(c)", li="100*log(i)")


@pytest.fixture(scope="module")
def bvar(data):
    return VAR(["ly", "lc", "li"], lags=4).fit(data, method=Bayes(prior=Minnesota(), draws=1000))


def test_marginal_likelihood_is_exact():
    """With one variable the marginal distribution of y is a multivariate t: compare with SciPy."""
    rng = np.random.default_rng(0)
    T = 30
    X = np.column_stack([np.ones(T), rng.normal(size=T)])
    Y = (X @ [0.5, 0.8] + rng.normal(size=T))[:, None]
    b, omega, Psi, d = np.array([[0.0], [0.3]]), np.array([4.0, 0.25]), np.array([[2.0]]), 5.0
    post = _niw(Y, X, b, omega, Psi, d)
    ref = stats.multivariate_t(loc=X @ b[:, 0], shape=Psi[0, 0] / d * (np.eye(T) + X @ np.diag(omega) @ X.T), df=d)
    assert post.logml == pytest.approx(ref.logpdf(Y[:, 0]), abs=1e-9)
    # and the posterior mean is the ridge-like estimate
    expected = np.linalg.solve(X.T @ X + np.diag(1 / omega), X.T @ Y + np.diag(1 / omega) @ b)
    assert np.allclose(post.B, expected)


def test_loose_prior_gives_least_squares(data):
    ols = VAR(["ly", "lc", "li"], lags=2).fit(data)
    loose = VAR(["ly", "lc", "li"], lags=2).fit(data, method=Bayes(prior=Minnesota(tightness=1e6, deterministic=1e12), draws=10))
    assert np.allclose(loose.estimate.B, ols.estimate.B, atol=1e-6)


def test_tight_prior_gives_the_prior_mean(data):
    tight = VAR(["ly", "lc"], lags=2).fit(data, method=Bayes(prior=Minnesota(tightness=1e-5, own_mean={"lc": 0.5}), draws=10))
    A = tight.coefs
    assert np.allclose(A[0], np.diag([1.0, 0.5]), atol=1e-4) and np.allclose(A[1], 0, atol=1e-4)


def test_posterior_draws_have_the_posterior_moments(bvar):
    assert isinstance(bvar, FittedBVAR) and bvar.draws_B.shape == (1000, 13, 3)
    post = bvar.posterior
    assert np.allclose(bvar.draws_B.mean(axis=0), post.B, atol=4 * np.sqrt(np.max(np.diag(post.Omega)) * np.max(bvar.estimate.sigma)) / np.sqrt(1000) + 1e-3)
    assert np.allclose(bvar.draws_sigma.mean(axis=0), bvar.estimate.sigma, rtol=0.1, atol=0.02)
    assert np.allclose(bvar.params.cov, np.kron(bvar.estimate.sigma, post.Omega))


def test_shrinkage_improves_on_least_squares_when_data_are_few(data):
    """Compared with least squares, the prior pulls coefficients towards the prior mean."""
    ols = VAR(["ly", "lc", "li"], lags=4).fit(data, span=(None, "2004Q4"))
    bvar = VAR(["ly", "lc", "li"], lags=4).fit(data, span=(None, "2004Q4"), method=Bayes(prior=Minnesota(tightness=0.1), draws=10))
    prior_mean = np.zeros_like(ols.coefs)
    prior_mean[0] = np.eye(3)
    assert np.abs(bvar.coefs - prior_mean).sum() < np.abs(ols.coefs - prior_mean).sum()


def test_dummy_priors_and_hyperparameter_choice(data, bvar):
    spec = VAR(["ly", "lc", "li"], lags=4)
    chosen = spec.fit(data, method=Bayes(prior=Minnesota(sum_of_coefficients=1, initial_observation=1, optimize=True), draws=50))
    assert set(chosen.hyperparameters) == {"tightness", "sum_of_coefficients", "initial_observation"}
    assert chosen.log_ml > bvar.log_ml  # the data prefer the chosen hyperparameters
    for factor in (0.8, 1.25):  # and it is a maximum
        h = dict(chosen.hyperparameters)
        h["tightness"] *= factor
        nearby = spec.fit(data, method=Bayes(prior=Minnesota(tightness=h["tightness"], sum_of_coefficients=h["sum_of_coefficients"],
                                                             initial_observation=h["initial_observation"]), draws=10))
        assert nearby.log_ml < chosen.log_ml + 1e-6
    with pytest.raises(ValueError, match="positive"):
        Minnesota(tightness=0)


def test_responses_and_forecasts_from_the_posterior(data, bvar):
    irf = bvar.irf(12)
    assert irf.values.shape == (13, 3, 3) and np.all(irf.lower <= irf.values) and np.all(irf.values <= irf.upper)
    assert "posterior median" in irf.identification
    point = bvar.irf(12, draws=0)
    assert point.lower is None and np.allclose(point.values[0], bvar.impact())
    fc = bvar.forecast(data, ("2025Q1", "2026Q4"))
    assert {"ly", "ly_se", "ly_median", "ly_lower", "ly_upper"} <= set(fc.names)
    widths = (fc["ly_upper"] - fc["ly_lower"]).to_numpy()
    assert np.all(np.diff(widths) > 0)  # uncertainty grows with the horizon
    ols_fc = VAR(["ly", "lc", "li"], lags=4).fit(data).forecast(data, ("2025Q1", "2026Q4"))
    assert abs(fc["ly"].iloc[0] - ols_fc["ly"].iloc[0]) < 2 * fc["ly_se"].iloc[0]
    assert "Bayesian VAR(4)" in str(bvar.summary()) and "log marginal likelihood" in str(bvar.summary())
    assert 0 <= bvar.explosive_share() <= 1
    assert bvar.fevd(8).values.shape[0] == 8


def test_dialog_records_and_replays(data, tmp_path):
    from pymetrix.commands import FitVAR, ImpulseResponse, Session, load_project, save_project

    s = Session()
    s.objects["data"] = data
    s.run(FitVAR("ly, lc", lags="2", bayesian=True, tightness=0.1, own_mean=1.0, sum_of_coefficients="1", draws=400))
    assert ("var = VAR(['ly', 'lc'], lags=2).fit(data, method=Bayes(prior=Minnesota(tightness=0.1, sum_of_coefficients=1.0), draws=400))"
            in s.script())
    s.run(ImpulseResponse("var", periods=8, draws=200))
    assert s["irf"].lower is not None
    assert FitVAR("ly, lc", bayesian=True, sum_of_coefficients="-1").validate(s) == ["Bayesian prior: sum_of_coefficients must be positive"]
    reopened = load_project(save_project(s, tmp_path / "bvar.pymx"))
    assert np.allclose(reopened["var"].draws_B, s["var"].draws_B)  # seeded draws: the same on replay
