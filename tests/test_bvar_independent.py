"""The independent normal-Wishart prior for VARs, checked independently: the Gibbs sampler's posterior means against
the exact posterior of an AR(1) (a one-dimensional integral over the error variance, the coefficients integrated in
closed form), and against the closed-form posterior given Sigma for a two-variable VAR whose Sigma the prior pins down;
the prior's variances as documented; cross-variable shrinkage; the dialog."""

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from pymetrix import VAR, Dataset, Minnesota, NormalWishart
from pymetrix.bvar import _gibbs
from pymetrix.estimation import Bayes


def _var_data(T=200, seed=0):
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.1, 0.0], [0.2, 0.4, 0.1], [0.0, 0.3, 0.6]])
    y = np.zeros((T, 3))
    for t in range(1, T):
        y[t] = A @ y[t - 1] + rng.normal(size=3)
    return Dataset.from_frame(pd.DataFrame(y, columns=["a", "b", "c"], index=pd.period_range("1970Q1", periods=T, freq="Q")))


def test_gibbs_matches_the_exact_posterior_of_an_ar1():
    rng = np.random.default_rng(1)
    T = 40
    y = np.zeros(T + 1)
    for t in range(1, T + 1):
        y[t] = 0.3 + 0.7 * y[t - 1] + rng.normal()
    Y, X = y[1:, None], np.column_stack([np.ones(T), y[:-1]])
    b, V, Psi, d = np.array([[0.0], [1.0]]), np.array([[4.0], [0.04]]), np.array([[1.5]]), 3.0
    V0 = np.diag(V[:, 0])

    def log_post(s2):  # p(s2 | y): the inverse-gamma prior times y ~ N(X b, s2 I + X V0 X')
        return stats.invgamma.logpdf(s2, d / 2, scale=Psi[0, 0] / 2) + stats.multivariate_normal.logpdf(
            Y[:, 0], X @ b[:, 0], s2 * np.eye(T) + X @ V0 @ X.T)

    grid = np.exp(np.linspace(np.log(0.2), np.log(5), 2000))
    lp = np.array([log_post(s) for s in grid])
    w = np.exp(lp - lp.max())
    cond = np.array([np.linalg.solve(np.linalg.inv(V0) + X.T @ X / s, np.linalg.solve(V0, b[:, 0]) + X.T @ Y[:, 0] / s) for s in grid])
    norm = integrate.trapezoid(w, grid)
    exact_b = integrate.trapezoid(w[:, None] * cond, grid, axis=0) / norm
    exact_s2 = integrate.trapezoid(w * grid, grid) / norm
    Bs, Ss = _gibbs(Y, X, b, V, Psi, d, 40_000, 1000, np.random.default_rng(0))
    assert Bs[:, :, 0].mean(axis=0) == pytest.approx(exact_b, abs=0.003)
    assert Ss[:, 0, 0].mean() == pytest.approx(exact_s2, abs=0.004)


def test_gibbs_given_a_known_sigma_is_the_closed_form_normal():
    rng = np.random.default_rng(2)
    T, K = 60, 2
    Sigma0 = np.array([[1.0, 0.6], [0.6, 2.0]])
    X = np.column_stack([np.ones(T), rng.normal(size=(T, 2))])
    Y = X @ np.array([[0.5, -0.2], [0.8, 0.1], [0.0, 0.4]]) + rng.multivariate_normal(np.zeros(2), Sigma0, size=T)
    b = np.zeros((3, K))
    V = np.array([[10.0, 10.0], [0.1, 0.02], [0.05, 0.3]])
    d = 1e7  # an inverse-Wishart this tight pins Sigma at Psi / d
    Bs, Ss = _gibbs(Y, X, b, V, Sigma0 * d, d, 4000, 200, np.random.default_rng(0))
    assert Ss.mean(axis=0) == pytest.approx(Sigma0, rel=1e-3)
    Si = np.linalg.inv(Sigma0)  # vec(B) | Sigma, equation by equation: precision diag(1/V) + Sigma^-1 (x) X'X
    P = np.diag(1 / V.reshape(-1, order="F")) + np.kron(Si, X.T @ X)
    mean = np.linalg.solve(P, (X.T @ Y @ Si).reshape(-1, order="F")).reshape(3, K, order="F")
    se = np.sqrt(np.diag(np.linalg.inv(P))).reshape(3, K, order="F")
    assert np.all(np.abs(Bs.mean(axis=0) - mean) < 4 * se / np.sqrt(4000))
    assert Bs.std(axis=0) == pytest.approx(se, rel=0.05)


def test_the_prior_shrinks_as_documented():
    data = _var_data()
    tight = VAR(["a", "b", "c"], lags=2).fit(data, method=Bayes(prior=NormalWishart(tightness=1e-3, own_mean=0.3), draws=500))
    lags = tight.estimate.B[1:4]
    assert np.allclose(np.diag(lags), 0.3, atol=0.01) and np.allclose(lags - np.diag(np.diag(lags)), 0, atol=0.01)
    sd = {}
    for theta in (1.0, 0.05):  # cross-variable lags shrink with cross_tightness; own lags hardly move
        fit = VAR(["a", "b", "c"], lags=1).fit(data, method=Bayes(prior=NormalWishart(own_mean=0, cross_tightness=theta), draws=2000))
        sd[theta] = np.sqrt(np.diag(fit.posterior.Omega)).reshape(fit.estimate.B.shape, order="F")[1:4]
        assert fit.params.values == pytest.approx(fit.estimate.B.T.reshape(-1))
    off = ~np.eye(3, dtype=bool)
    assert np.all(sd[0.05][off] < 0.5 * sd[1.0][off]) and np.allclose(np.diag(sd[0.05]), np.diag(sd[1.0]), rtol=0.15)
    loose = VAR(["a", "b", "c"], lags=2).fit(data, method=Bayes(prior=NormalWishart(own_mean=0, tightness=10, cross_tightness=1), draws=2000))
    ols = VAR(["a", "b", "c"], lags=2).fit(data)
    assert np.allclose(loose.estimate.B, ols.estimate.B, atol=0.01)  # a flat prior: least squares
    fit = VAR(["a", "b", "c"], lags=2).fit(data, method=Bayes(prior=NormalWishart(own_mean=0), draws=1000))
    assert "independent normal-Wishart" in str(fit.summary()) and np.isfinite(fit.log_ml)  # Chib's estimate
    bands = fit.irf(8).to_frame()
    assert len(bands) and fit.forecast(data, ("2020Q1", "2020Q4")).frame.notna().all().all()
    with pytest.raises(ValueError, match="positive"):
        NormalWishart(cross_tightness=0)


def test_the_var_dialog_offers_the_independent_prior():
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import FitVAR

    s = Session()
    s.objects["data"] = _var_data()
    cmd = FitVAR(variables="a,b,c", lags="1", bayesian=True, independent=True, cross_tightness=0.3, draws=300)
    assert isinstance(cmd._method().prior, NormalWishart) and cmd._method().prior.cross_tightness == 0.3
    assert FitVAR(variables="a,b", bayesian=True, independent=True, optimize=True)._method().prior.optimize  # by Chib's estimate
    assert FitVAR(variables="a,b", bayesian=True, independent=True, hierarchical=True).validate(s)
    assert isinstance(FitVAR(variables="a,b", bayesian=True)._method().prior, Minnesota)
    s.run(cmd)
    assert s.objects["var"].independent
