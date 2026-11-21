"""VARMA by conditional maximum likelihood, checked independently: with q = 0 it is the VAR's least squares; the
residuals by a plain loop and the log likelihood from them; a VARMA(1, 1) near statsmodels' VARMAX (exact maximum
likelihood) on a long sample; the moving-average representation by hand (Psi_1 = A_1 + M_1)."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import VAR, Dataset, varma
from pymetrix.errors import SpecificationError


@pytest.fixture(scope="module")
def simulated():
    rng = np.random.default_rng(0)
    T = 1500
    A, M = np.array([[0.5, 0.1], [0.2, 0.3]]), np.array([[0.4, 0.0], [0.1, -0.3]])
    e = rng.normal(size=(T, 2)) @ np.linalg.cholesky(np.array([[1, 0.3], [0.3, 1]])).T
    Y = np.zeros((T, 2))
    for t in range(1, T):
        Y[t] = np.array([0.1, -0.2]) + A @ Y[t - 1] + e[t] + M @ e[t - 1]
    return Dataset.from_frame(pd.DataFrame(Y, columns=["a", "b"], index=pd.period_range("1600Q1", periods=T, freq="Q"))), Y


def test_without_ma_terms_it_is_the_var(simulated):
    data, _ = simulated
    fit = varma(data, ["a", "b"], p=2, q=0)
    var = VAR(["a", "b"], lags=2).fit(data)
    assert fit.A == pytest.approx(var.coefs, abs=1e-10)


def test_the_residuals_and_statsmodels(simulated):
    data, Y = simulated
    fit = varma(data, ["a", "b"], p=1, q=1)
    c, A, M = fit.c, fit.A[0], fit.M[0]
    E, prev = [], np.zeros(2)
    for t in range(1, len(Y)):  # the residual recursion by a loop
        prev = Y[t] - c - A @ Y[t - 1] - M @ prev
        E.append(prev)
    E = np.array(E)
    assert fit.resid.to_numpy() == pytest.approx(E, abs=1e-10)
    n = len(E)
    assert fit.loglike == pytest.approx(-n / 2 * (np.linalg.slogdet(E.T @ E / n)[1] + 2 * (1 + np.log(2 * np.pi))), rel=1e-10)
    from statsmodels.tsa.statespace.varmax import VARMAX

    ref = VARMAX(Y, order=(1, 1), trend="c").fit(disp=False, maxiter=1000)
    theirs = np.asarray(ref.params)[:10]  # constants, then A by rows, then M by rows
    assert fit.values == pytest.approx(theirs, abs=0.04)  # conditional and exact likelihoods agree in long samples
    Psi = fit.ma_representation(2)
    assert Psi[1] == pytest.approx(A + M, abs=1e-12) and Psi[2] == pytest.approx(A @ (A + M), abs=1e-12)
    assert fit.irf(8).values.shape == (9, 2, 2) and any("invertible" in n for n in fit.summary().notes)


def test_refusals(simulated):
    data, _ = simulated
    with pytest.raises(SpecificationError, match="not both zero"):
        varma(data, ["a", "b"], p=0, q=0)
