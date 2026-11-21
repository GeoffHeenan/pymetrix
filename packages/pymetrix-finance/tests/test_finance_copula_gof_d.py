"""The goodness-of-fit test of the Gaussian and t copulas for more than two series, checked independently:

- the d-dimensional CDF (Genz's separation of variables by quasi-Monte Carlo) against R's mvtnorm 1.x
  (pmvnorm and pmvt, GenzBretz with abseps 1e-9, the copula package's C.n for the empirical copula; numbers written
  here): 2e-5 at the default 1,024 points and 2e-6 at 65,536; against scipy's multivariate normal CDF in four
  dimensions (1e-4); in two dimensions against the module's own bivariate quadrature (the t: 5e-5);
- the Cramer-von Mises statistic S_n of three-series data at given parameters against the same R computation
  (relative 2e-3);
- the parametric-bootstrap test's size on data from the null (a small seeded study here: 30 replications, at most 5
  rejections at 5%; 200 with PYMETRIX_SLOW=1), its power against a wrong family (exchangeable Clayton data tested as
  Gaussian), and the p-value reproducible with its seed; through ``copula(..., gof=True)`` for three series.
"""

import importlib
import os

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pymetrix import Dataset, copula

cp = importlib.import_module("pymetrix_finance.copulas")

R0 = np.array([[1, 0.45, 0.3], [0.45, 1, 0.35], [0.3, 0.35, 1]])
R_GAUSS = [0.2517300590, 0.0251190934, 0.3599703144, 0.2150983948]  # pmvnorm at the first four rows
R_T5 = [0.2494924664, 0.0293156677, 0.3590637787, 0.2100148944]  # pmvt, df = 5
R_SN = (0.0577250950, 0.0522574533)  # sum((C.n(P, P) - C(P))^2), Gaussian and t5 at R0


def _sample():
    return cp.pseudo_observations(cp.sample_copula("gaussian", {"R": R0}, 150, 3, np.random.default_rng(7)))


def test_cdf_against_r_and_scipy():
    P = _sample()
    assert cp.elliptical_cdf_d(P[:4], R0) == pytest.approx(R_GAUSS, abs=2e-5)
    assert cp.elliptical_cdf_d(P[:4], R0, 5.0) == pytest.approx(R_T5, abs=2e-5)
    assert cp.elliptical_cdf_d(P[:4], R0, None, 65536) == pytest.approx(R_GAUSS, abs=2e-6)
    assert cp.elliptical_cdf_d(P[:4], R0, 5.0, 65536) == pytest.approx(R_T5, abs=2e-6)
    R4 = np.array([[1, 0.5, 0.3, 0.2], [0.5, 1, 0.4, 0.1], [0.3, 0.4, 1, -0.2], [0.2, 0.1, -0.2, 1]])
    U = cp.sample_copula("gaussian", {"R": R4}, 200, 4, np.random.default_rng(1))
    exact = stats.multivariate_normal(np.zeros(4), R4).cdf(stats.norm.ppf(U))
    assert cp.copula_cdf("gaussian", {"R": R4}, U) == pytest.approx(exact, abs=1e-4)
    two = cp.elliptical_cdf_d(U[:, :2], R4[:2, :2], 6.0)
    assert two == pytest.approx(cp._elliptical_cdf(U[:, 0], U[:, 1], 0.5, 6.0), abs=5e-5)


def test_statistic_against_r():
    P = _sample()
    assert cp.cramer_von_mises(P, "gaussian", {"R": R0}) == pytest.approx(R_SN[0], rel=2e-3)
    assert cp.cramer_von_mises(P, "t", {"R": R0, "nu": 5.0}) == pytest.approx(R_SN[1], rel=2e-3)


def _size(family, params, reps, n_boot, n=100):
    rejections = 0
    for i in range(reps):
        U = cp.sample_copula(family, params, n, 3, np.random.default_rng(2000 + i))
        rejections += cp.gof_test(cp.pseudo_observations(U), family, "ml", n_boot, seed=i)["pvalue"] < 0.05
    return rejections


def test_size_power_and_seed():
    assert _size("gaussian", {"R": R0}, 30, 30) <= 5  # P(Binomial(30, 0.05) > 5) = 0.003
    U = cp.sample_copula("clayton", {"theta": 3.0}, 250, 3, np.random.default_rng(3))
    P = cp.pseudo_observations(U)
    assert cp.gof_test(P, "gaussian", "ml", 50, seed=1)["pvalue"] < 0.05
    idx = pd.period_range("2000-01-01", periods=250, freq="D", name="time")
    data = Dataset.from_frame(pd.DataFrame(stats.norm.ppf(U), columns=["a", "b", "c"], index=idx))
    fit = copula(data, ["a", "b", "c"], family="t", gof=True, n_boot=30, seed=4)
    assert 0 < fit.gof_result["pvalue"] <= 1 and fit.goodness_of_fit(30, seed=4) == fit.gof_result
    assert "goodness of fit p-value" in str(fit.summary())


@pytest.mark.skipif(not os.environ.get("PYMETRIX_SLOW"), reason="several minutes: set PYMETRIX_SLOW=1")
def test_size_200_replications():
    assert 2 <= _size("gaussian", {"R": R0}, 200, 100) <= 19  # Binomial(200, 0.05): 99% band [3, 18]
    assert 2 <= _size("t", {"R": R0, "nu": 6.0}, 200, 100) <= 19
