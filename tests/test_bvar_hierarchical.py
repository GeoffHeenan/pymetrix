"""Hierarchical (Giannone-Lenza-Primiceri) priors for Bayesian VARs and Chib's marginal likelihood for the independent
normal-Wishart prior, checked independently: GLP's eigenvalue formula against the conjugate marginal likelihood at fixed
hyperparameters (with and without the dummy observations); the gamma hyperpriors' modes and standard deviations; the
mode a maximum; the hyperparameters' MCMC against their posterior on a grid; Chib's estimate against quadrature (one
variable, with and without dummies) and against importance sampling (two variables); the hyperparameter search; the
dialog."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from pymetrix import VAR, Bayes, Dataset, GLP, NormalWishart
from pymetrix.bvar import _niw, _posterior, _setup
from pymetrix.bvar_hierarchical import _glp_ml, _log_post, _space, chib_log_ml, gamma_shape_scale, glp_log_ml


def _var_data(T=120, K=3, seed=0):
    rng = np.random.default_rng(seed)
    A = np.array([[0.9, 0.1, 0.0], [0.05, 0.8, 0.1], [0.0, 0.2, 0.7]])[:K, :K]
    y = np.zeros((T, K))
    for t in range(1, T):
        y[t] = 0.5 + A @ y[t - 1] + rng.normal(size=K)
    cols = ["a", "b", "c"][:K]
    return Dataset.from_frame(pd.DataFrame(y + 10, columns=cols, index=pd.period_range("1970Q1", periods=T, freq="Q")))


def test_glps_formula_is_the_conjugate_marginal_likelihood():
    rng = np.random.default_rng(0)
    for T, K, k in ((40, 3, 7), (12, 2, 5), (60, 1, 3)):
        X, Y = rng.normal(size=(T, k)), rng.normal(size=(T, K))
        b, omega, psi = rng.normal(size=(k, K)), rng.uniform(0.1, 2, k), rng.uniform(0.5, 2, K)
        assert glp_log_ml(Y, X, b, omega, psi, K + 2) == pytest.approx(_niw(Y, X, b, omega, np.diag(psi), K + 2).logml, abs=1e-9)


@pytest.fixture(scope="module")
def three():
    data = _var_data()
    fit = VAR(["a", "b", "c"], lags=2).fit(data, method=Bayes(prior=GLP(), draws=300))
    return data, fit


def test_with_the_dummies_too_at_fixed_hyperparameters(three):
    _, fit = three
    prior = GLP()
    s = _setup(fit.spec, fit.arrays, fit.lags, prior)
    space = _space(s, prior)
    h = np.array([0.3, 2.0, 0.7, *(s.psi * 1.3)])
    conjugate = _posterior(replace(s, psi=h[3:]), prior, 0.3, 2.0, 0.7).logml
    assert _glp_ml(s, prior, h, space) == pytest.approx(conjugate, abs=1e-8)


def test_the_hyperpriors_have_their_stated_modes_and_sds():
    for mode, sd in ((0.2, 0.4), (1.0, 1.0)):
        k, theta = gamma_shape_scale(mode, sd)
        assert (k - 1) * theta == pytest.approx(mode) and np.sqrt(k) * theta == pytest.approx(sd)


def test_the_mode_is_a_maximum(three):
    _, fit = three
    prior = GLP()
    s = _setup(fit.spec, fit.arrays, fit.lags, prior)
    space = _space(s, prior)
    h = np.array(list(fit.hyperparameters.values()))
    top = _log_post(s, prior, h, space)
    assert top == pytest.approx(fit.provenance.method["hyperparameter log posterior at the mode"], abs=1e-5)
    for j in range(len(h)):
        for step in (0.98, 1.02):
            assert _log_post(s, prior, h * np.where(np.arange(len(h)) == j, step, 1.0), space) < top + 1e-7
    assert fit.log_ml == pytest.approx(_glp_ml(s, prior, h, space), abs=1e-8)
    assert list(fit.hyperparameters)[:3] == ["tightness", "sum_of_coefficients", "initial_observation"]
    assert "Giannone, Lenza and Primiceri" in str(fit.summary()) and fit.hyperparameter_draws is None


def test_mcmc_over_the_hyperparameters_matches_their_posterior_on_a_grid():
    data = _var_data(T=80, K=1, seed=3)
    prior = GLP(mcmc=True, sum_of_coefficients=False, initial_observation=False, burn=600)
    fit = VAR(["a"], lags=2).fit(data, method=Bayes(prior=prior, draws=4000, seed=2))
    s = _setup(fit.spec, fit.arrays, fit.lags, prior)
    space = _space(s, prior)
    lam = np.exp(np.linspace(np.log(0.02), np.log(4.9), 110))
    psi = np.exp(np.linspace(np.log(space.lows[1] * 1.001), np.log(space.highs[1] * 0.999), 110))
    lp = np.array([[_log_post(s, prior, np.array([a, p]), space) for p in psi] for a in lam])
    w = np.exp(lp - lp.max())
    z = integrate.trapezoid(integrate.trapezoid(w, psi, axis=1), lam)
    m_lam = integrate.trapezoid(integrate.trapezoid(w, psi, axis=1) * lam, lam) / z
    m_psi = integrate.trapezoid(integrate.trapezoid(w * psi[None], psi, axis=1), lam) / z
    d = fit.hyperparameter_draws
    from pymetrix.models.bayes import ess

    for col, exact in (("tightness", m_lam), ("psi a", m_psi)):
        x = d[col].to_numpy()
        assert abs(x.mean() - exact) < 4 * x.std() / np.sqrt(ess(x[None]))
    assert 0.15 < fit.provenance.method["acceptance"] < 0.45
    assert fit.from_draws and fit.draws_B.shape[0] == 4000
    assert np.allclose(fit.params.values, fit.draws_B.mean(axis=0).T.reshape(-1))
    assert fit.forecast(data, ("1990Q1", "1990Q4")).frame.notna().all().all()


def _exact_one(Y, X, b, V, Psi, d):
    """log p(Y) for one variable: the inverse gamma on sigma^2 by quadrature, the coefficients integrated in closed form"""
    T = len(Y)
    V0 = np.diag(V[:, 0])

    def lp(s2):
        return stats.invgamma.logpdf(s2, d / 2, scale=Psi[0, 0] / 2) + stats.multivariate_normal.logpdf(
            Y[:, 0], X @ b[:, 0], s2 * np.eye(T) + X @ V0 @ X.T)

    g = np.exp(np.linspace(np.log(0.02), np.log(50), 800))
    v = np.array([lp(s) for s in g])
    return v.max() + np.log(integrate.trapezoid(np.exp(v - v.max()), g))


def test_chibs_estimate_against_quadrature():
    rng = np.random.default_rng(1)
    T = 40
    y = np.zeros(T + 1)
    for t in range(1, T + 1):
        y[t] = 0.3 + 0.7 * y[t - 1] + rng.normal()
    Y, X = y[1:, None], np.column_stack([np.ones(T), y[:-1]])
    b, V, Psi, d = np.array([[0.0], [1.0]]), np.array([[4.0], [0.04]]), np.array([[1.5]]), 3.0
    exact = _exact_one(Y, X, b, V, Psi, d)
    assert chib_log_ml(Y, X, b, V, Psi, d, 4000, 300, np.random.default_rng(0)) == pytest.approx(exact, abs=0.01)
    Yd, Xd = np.array([[2.0]]), np.array([[1.0, 2.0]])  # a dummy observation: p(Y) = p(Y, Yd) / p(Yd)
    exact = _exact_one(np.vstack([Yd, Y]), np.vstack([Xd, X]), b, V, Psi, d) - _exact_one(Yd, Xd, b, V, Psi, d)
    assert chib_log_ml(Y, X, b, V, Psi, d, 4000, 300, np.random.default_rng(0), (Yd, Xd)) == pytest.approx(exact, abs=0.02)


def test_chibs_estimate_against_importance_sampling():
    rng = np.random.default_rng(1)
    T, K = 10, 2
    X = np.column_stack([np.ones(T), rng.normal(size=T)])
    Y = X @ np.array([[0.5, -0.2], [0.8, 0.1]]) + rng.normal(size=(T, K))
    b, V, Psi, d = np.zeros((2, K)), np.array([[1.0, 1.0], [0.5, 0.5]]), np.eye(2) * 2, 6.0
    N = 200_000  # draws from the prior, weighted by the likelihood
    r = np.random.default_rng(5)
    S = stats.invwishart.rvs(df=d, scale=Psi, size=N, random_state=r)
    B = b[None] + np.sqrt(V)[None] * r.standard_normal((N, 2, K))
    E = Y[None] - np.einsum("tk,nkj->ntj", X, B)
    ll = -T * K / 2 * np.log(2 * np.pi) - T / 2 * np.linalg.slogdet(S)[1] - 0.5 * np.einsum("nij,nti,ntj->n", np.linalg.inv(S), E, E)
    w = np.exp(ll - ll.max())
    estimate, se = ll.max() + np.log(w.mean()), w.std() / w.mean() / np.sqrt(N)
    assert chib_log_ml(Y, X, b, V, Psi, d, 4000, 300, np.random.default_rng(0)) == pytest.approx(estimate, abs=4 * se + 0.01)


def test_the_independent_priors_search():
    data = _var_data(T=100, K=2, seed=4)
    method = Bayes(prior=NormalWishart(own_mean=1.0, tightness=0.05, cross_tightness=0.9, burn=200, optimize=True), draws=300, seed=1)
    fit = VAR(["a", "b"], lags=1).fit(data, method=method)
    fixed = VAR(["a", "b"], lags=1).fit(data, method=replace(method, prior=replace(method.prior, optimize=False)))
    assert np.isfinite(fixed.log_ml) and fit.log_ml > fixed.log_ml
    assert fit.hyperparameters["tightness"] != 0.05 and "Nelder-Mead" in fit.provenance.method["search"]
    assert "chosen by marginal likelihood" in str(fit.summary())


def test_the_posterior_updated_by_conditions_uses_the_fitted_prior(monkeypatch):
    """Waggoner and Zha's sampler (update_posterior=True) draws the parameters under the prior that was fitted: the
    hyperparameters the search chose for the independent prior, GLP's estimated psi (it used the starting values and
    the autoregressions' psi)."""
    from pymetrix import bvar as bv
    from pymetrix.bvar import _independent_prior, _prior

    data = _var_data(T=100, K=2, seed=4)
    span, condition = ("1995Q1", "1995Q2"), {"a": 10.0}
    nw = VAR(["a", "b"], lags=1).fit(data, method=Bayes(prior=NormalWishart(tightness=0.05, cross_tightness=0.9, burn=100, optimize=True),
                                                        draws=50, seed=1))
    seen: dict = {}
    step, niw = bv._gibbs_step, bv._niw

    def spy_step(Y, X, b, V, *rest):
        seen.setdefault("V", V)
        return step(Y, X, b, V, *rest)

    def spy_niw(Y, X, b, omega, Psi, d):
        seen.setdefault("glp", (omega, Psi))
        return niw(Y, X, b, omega, Psi, d)

    monkeypatch.setattr(bv, "_gibbs_step", spy_step)
    nw.conditional_forecast(data, span, condition, draws=3, update_posterior=True)
    h = nw.hyperparameters
    fitted = replace(nw.method.prior, tightness=h["tightness"], cross_tightness=h["cross tightness"])
    assert h["tightness"] != 0.05
    np.testing.assert_allclose(seen["V"], _independent_prior(_setup(nw.spec, nw.arrays, 1, fitted), fitted)[1])

    glp = VAR(["a", "b"], lags=1).fit(data, method=Bayes(prior=GLP(sum_of_coefficients=False, initial_observation=False), draws=50, seed=1))
    monkeypatch.setattr(bv, "_niw", spy_niw)
    glp.conditional_forecast(data, span, condition, draws=3, update_posterior=True)
    h = glp.hyperparameters
    s = replace(_setup(glp.spec, glp.arrays, 1, glp.method.prior), psi=np.array([h["psi a"], h["psi b"]]))
    _, omega, Psi, _, _ = _prior(s, glp.method.prior, h["tightness"], None, None)
    np.testing.assert_allclose(seen["glp"][0], omega)
    np.testing.assert_allclose(seen["glp"][1], Psi)


def test_the_dialog(three):
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import FitVAR

    data, _ = three
    s = Session()
    s.objects["data"] = data
    s.run(FitVAR("a, b", lags="1", bayesian=True, hierarchical=True, hyper_mcmc=True, draws=200))
    assert isinstance(s["var"].method.prior, GLP) and s["var"].hyperparameter_draws is not None
    assert "prior=GLP(mcmc=True)" in s.script()
