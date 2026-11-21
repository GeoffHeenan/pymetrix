"""Generalised impulse responses, GARCH-in-mean and the neural-network test, checked independently: each shock's
generalised response equal to the Cholesky response with that variable ordered first (Pesaran and Shin's identity),
and invariant to the VAR's ordering; GARCH-in-mean without the in-mean term equal to arch's constant-mean GARCH(1, 1)
(estimates, log likelihood, robust standard errors), and the in-mean coefficient of a simulated model recovered; the
neural-network test's F by hand from its hidden units, its size under a linear AR and its power against a nonlinear one."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import VAR, Cholesky, Dataset, Equation, garch_in_mean


def _var_data(T=300, seed=0):
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.1, 0.0], [0.2, 0.4, 0.1], [0.0, 0.2, 0.3]])
    C = np.linalg.cholesky(np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]]))
    y = np.zeros((T, 3))
    for t in range(1, T):
        y[t] = A @ y[t - 1] + C @ rng.normal(size=3)
    return Dataset.from_frame(pd.DataFrame(y, columns=["a", "b", "c"], index=pd.period_range("1900Q1", periods=T, freq="Q")))


def test_generalised_responses_are_cholesky_with_each_variable_first():
    data = _var_data()
    var = VAR(["a", "b", "c"], lags=1).fit(data)
    g = var.girf(12, draws=0)
    for j, v in enumerate(["a", "b", "c"]):
        order = [v] + [w for w in ["a", "b", "c"] if w != v]
        chol = var.identify(Cholesky(order)).irf(12, draws=0)
        assert g.values[:, :, j] == pytest.approx(chol.values[:, :, j], abs=1e-12)
    other = VAR(["c", "a", "b"], lags=1).fit(data).girf(12, draws=0)  # the ordering does not matter
    assert g.values[:, 0, 0] == pytest.approx(other.values[:, 1, 1], abs=1e-10)
    banded = var.girf(8, draws=50)
    assert banded.lower.shape == banded.values.shape and np.all(banded.lower <= banded.upper)


def _garch_m(T=2000, seed=0, delta=0.1):
    rng = np.random.default_rng(seed)
    h, r, e = np.empty(T), np.empty(T), np.empty(T)
    h[0] = 1
    for t in range(T):
        if t:
            h[t] = 0.05 + 0.1 * e[t - 1] ** 2 + 0.85 * h[t - 1]
        e[t] = np.sqrt(h[t]) * rng.normal()
        r[t] = 0.02 + delta * h[t] + e[t]
    return Dataset.from_frame(pd.DataFrame({"r": r}, index=pd.period_range("2000-01-01", periods=T, freq="D"))), r


def test_garch_without_the_in_mean_term_is_arch():
    arch = pytest.importorskip("arch")
    data, r = _garch_m()
    fit = garch_in_mean(data, "r", form=None)
    ref = arch.arch_model(r, mean="Constant", vol="GARCH", p=1, q=1).fit(disp="off", options={"ftol": 1e-12})
    assert fit.values == pytest.approx(ref.params.to_numpy(), rel=1e-5)
    assert fit.loglike == pytest.approx(ref.loglikelihood, rel=1e-10)
    assert fit.se == pytest.approx(ref.std_err.to_numpy(), rel=2e-3)


def test_the_in_mean_coefficient_is_recovered():
    data, _ = _garch_m(T=3000, seed=1)
    fit = garch_in_mean(data, "r")
    assert fit.params["delta"] == pytest.approx(0.1, abs=3 * fit.se[4]) and fit.names[-1] == "delta"
    assert garch_in_mean(data, "r", form="sd").names[-1] == "delta"


def _ar(T, seed, nonlinear):
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = (0.9 * np.tanh(2 * y[t - 1]) if nonlinear else 0.5 * y[t - 1]) + 0.5 * rng.normal()
    return Dataset.from_frame(pd.DataFrame({"y": y}, index=pd.period_range("1900Q1", periods=T, freq="Q")))


def test_the_neural_network_test():
    data = _ar(300, 0, nonlinear=False)
    fit = Equation("y ~ y[-1]").fit(data)
    res = fit.test("neural_network", seed=3)
    y = data.frame["y"].to_numpy()
    X = np.column_stack([np.ones(299), y[:-1]])
    u = y[1:] - X @ np.linalg.lstsq(X, y[1:], rcond=None)[0]
    w = (y[:-1] - y[:-1].mean()) / y[:-1].std()
    gamma = np.random.default_rng(3).uniform(-2, 2, (2, 10))
    H = 1 / (1 + np.exp(-np.column_stack([np.ones(299), w]) @ gamma))
    H -= H.mean(axis=0)
    U, s, _ = np.linalg.svd(H, full_matrices=False)
    full = np.column_stack([X, U[:, :2] * s[:2]])
    e = y[1:] - full @ np.linalg.lstsq(full, y[1:], rcond=None)[0]
    F = ((u @ u - e @ e) / 2) / (e @ e / (299 - 4))
    assert res.statistic == pytest.approx(F, rel=1e-10) and res.df == (2, 295)
    size = np.mean([Equation("y ~ y[-1]").fit(_ar(300, 10 + k, False)).test("neural_network", seed=k).p_value < 0.05 for k in range(100)])
    power = np.mean([Equation("y ~ y[-1]").fit(_ar(300, 200 + k, True)).test("neural_network", seed=k).p_value < 0.05 for k in range(40)])
    assert size == pytest.approx(0.05, abs=0.05) and power > 0.7
