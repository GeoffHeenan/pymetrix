"""Threshold VARs, bilinear models and BHHH, checked independently: the threshold VAR's regimes by least squares on
the periods at or below the threshold, its threshold the brute-force minimiser of the pooled log determinant, its LR
statistic by hand, a simulated threshold recovered and the bootstrap test rejecting it but not a linear VAR; the
bilinear model's residual recursion by a loop and its parameters recovered (b near zero on an AR(1), whose phi then
matches least squares); BHHH reaching Newton's estimates, with OPG standard errors by hand."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import ML, Dataset, Equation, Logit, bilinear, threshold_var


def _tvar_data(T=600, seed=0, linear=False):
    rng = np.random.default_rng(seed)
    A1, A2 = np.array([[0.7, 0.1], [0.0, 0.5]]), np.array([[-0.3, 0.2], [0.3, 0.2]])
    Y = np.zeros((T, 2))
    for t in range(1, T):
        A, c = (A1, np.array([0.5, 0])) if (linear or Y[t - 1, 0] <= 0.2) else (A2, np.array([-0.5, 0.3]))
        Y[t] = c + A @ Y[t - 1] + rng.normal(size=2) * 0.5
    return Dataset.from_frame(pd.DataFrame(Y, columns=["a", "b"], index=pd.period_range("1900Q1", periods=T, freq="Q"))), Y


def test_the_threshold_var_by_hand():
    data, Y = _tvar_data()
    fit = threshold_var(data, ["a", "b"], lags=1)
    y, X, z = Y[1:], np.column_stack([np.ones(len(Y) - 1), Y[:-1]]), Y[:-1, 0]

    def pooled(g):
        low = z <= g
        U = np.empty_like(y)
        for m in (low, ~low):
            U[m] = y[m] - X[m] @ np.linalg.lstsq(X[m], y[m], rcond=None)[0]
        return np.linalg.slogdet(U.T @ U / len(U))[1]

    s = np.sort(z)
    grid = np.unique(s[int(np.floor(0.15 * len(z))):int(np.ceil(0.85 * len(z)))])
    best = grid[np.argmin([pooled(g) for g in grid])]
    assert fit.gamma == best and fit.gamma == pytest.approx(0.2, abs=0.1)
    low = z <= fit.gamma
    assert fit.coefs_low == pytest.approx(np.linalg.lstsq(X[low], y[low], rcond=None)[0], abs=1e-12)
    e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    assert fit.LR == pytest.approx(len(y) * (np.linalg.slogdet(e.T @ e / len(y))[1] - pooled(fit.gamma)), rel=1e-10)
    assert fit.test(draws=60, seed=1).p_value < 0.05
    assert fit.regime_table().title.startswith("Threshold VAR")


def test_a_linear_var_is_not_rejected():
    data, _ = _tvar_data(T=400, seed=3, linear=True)
    assert threshold_var(data, ["a", "b"], lags=1).test(draws=60, seed=2).p_value > 0.05


def test_bilinear():
    rng = np.random.default_rng(4)
    T = 1000
    e = rng.normal(size=T) * 0.5
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.2 + 0.4 * y[t - 1] + 0.3 * y[t - 1] * e[t - 1] + e[t]
    data = Dataset.from_frame(pd.DataFrame({"y": y}, index=pd.period_range("1900-01", periods=T, freq="M")))
    fit = bilinear(data, "y")
    for v, true, s in zip(fit.values, (0.2, 0.4, 0.3), fit.se):
        assert v == pytest.approx(true, abs=3 * s)
    mu, phi, b = fit.values
    r, prev = [], 0.0
    for t in range(1, T):  # the residuals by a loop
        prev = y[t] - mu - phi * y[t - 1] - b * y[t - 1] * prev
        r.append(prev)
    r = np.array(r)
    assert fit.loglike == pytest.approx(-len(r) / 2 * (np.log(2 * np.pi) + np.log(r @ r / len(r)) + 1), rel=1e-10)
    ar = np.zeros(T)
    for t in range(1, T):
        ar[t] = 0.2 + 0.5 * ar[t - 1] + e[t]
    plain = bilinear(Dataset.from_frame(pd.DataFrame({"y": ar}, index=data.frame.index)), "y")
    assert plain.values[2] == pytest.approx(0, abs=3 * plain.se[2])
    assert plain.values[1] == pytest.approx(np.polyfit(ar[:-1], ar[1:], 1)[0], abs=0.02)


def test_bhhh_reaches_newtons_estimates():
    rng = np.random.default_rng(5)
    n = 800
    x = rng.normal(size=n)
    d = (0.3 + 0.8 * x + rng.logistic(size=n) > 0).astype(float)
    data = Dataset.from_frame(pd.DataFrame({"id": range(n), "d": d, "x": x}), entity="id")
    newton = Equation("d ~ x", family=Logit()).fit(data, method=ML())
    bhhh = Equation("d ~ x", family=Logit()).fit(data, method=ML(optimizer="bhhh", max_iter=500))
    assert bhhh.params.values == pytest.approx(newton.params.values, abs=1e-7)
    opg = Equation("d ~ x", family=Logit()).fit(data, method=ML(cov="opg"))
    X = np.column_stack([np.ones(n), x])
    p = 1 / (1 + np.exp(-X @ newton.params.values))
    G = X * (d - p)[:, None]
    assert opg.params.se == pytest.approx(np.sqrt(np.diag(np.linalg.inv(G.T @ G))), rel=1e-6)
