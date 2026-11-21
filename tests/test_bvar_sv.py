"""BVARs with stochastic volatility, checked independently: the coefficients' conditional posterior with a constant
error covariance and a diffuse prior equal to least squares (as SUR with common regressors must be), and its
covariance to the SUR formula Sigma (x) (X'X)^-1; a simulated volatility break recovered in one shock and absent in
the other; the posterior mean coefficients near least squares under a loose prior; responses at a date from the kept
draws by hand."""

import importlib

import numpy as np
import pandas as pd
import pytest

from pymetrix import BVARSV, Dataset
from pymetrix.errors import SpecificationError

bsv = importlib.import_module("pymetrix.bvar_sv")


def _data(T=300, seed=0, break_at=150):
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.1], [0.2, 0.4]])
    y = np.zeros((T, 2))
    for t in range(1, T):
        sd = np.array([1.0, 1.0]) if t < break_at else np.array([3.0, 1.0])
        y[t] = A @ y[t - 1] + sd * rng.normal(size=2)
    return Dataset.from_frame(pd.DataFrame(y, columns=["a", "b"], index=pd.period_range("1900Q1", periods=T, freq="Q"))), y


def test_the_coefficient_step_is_sur_least_squares():
    _, y = _data()
    Y, X = bsv._design(y, 2)
    S = np.array([[1.0, 0.3], [0.3, 2.0]])
    mean, cov = bsv.coefficient_posterior(Y, X, np.tile(np.linalg.inv(S), (len(Y), 1, 1)), np.zeros(10), np.eye(10) * 1e-12)
    assert mean == pytest.approx(np.linalg.lstsq(X, Y, rcond=None)[0].T.reshape(-1), abs=1e-10)
    assert cov == pytest.approx(np.kron(S, np.linalg.inv(X.T @ X)), rel=1e-6)


@pytest.fixture(scope="module")
def fitted():
    data, y = _data()
    return BVARSV(["a", "b"], lags=1).fit(data, draws=400, burn=300, tightness=1.0, seed=1), y


def test_the_volatility_break_is_found(fitted):
    fit, _ = fitted
    v = fit.volatilities()
    a_early, a_late = v["a"].iloc[:140].median(), v["a"].iloc[160:].median()
    b_early, b_late = v["b"].iloc[:140].median(), v["b"].iloc[160:].median()
    assert a_early == pytest.approx(1.0, abs=0.25) and a_late == pytest.approx(3.0, abs=0.6)
    assert 0.7 < b_late / b_early < 1.4


def test_coefficients_and_responses(fitted):
    fit, y = fitted
    Y, X = bsv._design(y, 1)
    ols = np.linalg.lstsq(X, Y, rcond=None)[0][1:].T  # the lag matrix
    assert fit.coefs[0] == pytest.approx(ols, abs=0.08)  # a loose prior: near least squares (GLS weights the volatile shock less)
    t = 200
    irf = fit.irf(4, period=str(fit.index[t]))
    from pymetrix.var import _ma

    draws = [_ma(np.stack([b[:, 1:3]]), 4) @ (np.linalg.inv(a) * np.exp(h[t])) for b, a, h in zip(fit.B, fit.A, fit.h)]
    assert irf.values == pytest.approx(np.median(np.array(draws), axis=0), abs=1e-12)
    with pytest.raises(SpecificationError, match="lag"):
        BVARSV(["a"], lags=0)
