"""Count and duration time series, checked independently: INAR(1)'s transition probabilities by explicit summation
and its simulated parameters recovered; the Poisson autoregression's likelihood by a loop and its parameters
recovered; the exponential ACD equal to Gaussian GARCH(1, 1) quasi-likelihood on the square-root durations (Engle and
Russell's equivalence) and near arch's GARCH fit, the Weibull ACD's shape near 1 for exponential data and its
likelihood by scipy's Weibull density; and the dynamic probit as the probit with a lagged dependent variable
against statsmodels."""

import numpy as np
import pandas as pd
import pytest
from scipy import special
from scipy import stats as sps

from pymetrix import Dataset, Equation, ML, Probit, acd, inar, poisson_autoregression
from pymetrix.errors import SpecificationError


def _ds(values):
    return Dataset.from_frame(pd.DataFrame({"y": values}, index=pd.period_range("1900-01", periods=len(values), freq="M")))


@pytest.fixture(scope="module")
def counts():
    rng = np.random.default_rng(0)
    y = np.zeros(1000, int)
    y[0] = 5
    for t in range(1, 1000):
        y[t] = rng.binomial(y[t - 1], 0.6) + rng.poisson(2.0)
    return y


def test_inar(counts):
    fit = inar(_ds(counts), "y")
    alpha, lam = fit.values
    assert alpha == pytest.approx(0.6, abs=3 * fit.se[0]) and lam == pytest.approx(2.0, abs=3 * fit.se[1])
    ll = 0.0
    for prev, cur in zip(counts[:-1], counts[1:]):  # explicit summation over the survivors
        ll += np.log(sum(sps.binom.pmf(k, prev, alpha) * sps.poisson.pmf(cur - k, lam) for k in range(min(prev, cur) + 1)))
    assert fit.loglike == pytest.approx(ll, rel=1e-10)
    with pytest.raises(SpecificationError, match="whole numbers"):
        inar(_ds(counts + 0.5), "y")


def test_poisson_autoregression():
    rng = np.random.default_rng(1)
    T = 1000
    lam, y = np.zeros(T), np.zeros(T, int)
    lam[0] = 3
    for t in range(T):
        if t:
            lam[t] = 0.5 + 0.3 * y[t - 1] + 0.5 * lam[t - 1]
        y[t] = rng.poisson(lam[t])
    fit = poisson_autoregression(_ds(y), "y")
    for v, true, s in zip(fit.values, (0.5, 0.3, 0.5), fit.se):
        assert v == pytest.approx(true, abs=3 * s)
    omega, a, b = fit.values
    path, ll = [y.mean()], 0.0
    for t in range(1, T):
        path.append(omega + a * y[t - 1] + b * path[-1])
        ll += sps.poisson.logpmf(y[t], path[-1])
    assert fit.loglike == pytest.approx(ll, rel=1e-10) and fit.fitted.to_numpy() == pytest.approx(path, rel=1e-10)
    plain = poisson_autoregression(_ds(y), "y", feedback=False)
    assert plain.names == ("omega", "a") and plain.loglike < fit.loglike


@pytest.fixture(scope="module")
def durations():
    rng = np.random.default_rng(2)
    T = 1000
    psi, x = np.zeros(T), np.zeros(T)
    psi[0] = 1
    for t in range(T):
        if t:
            psi[t] = 0.1 + 0.15 * x[t - 1] + 0.75 * psi[t - 1]
        x[t] = psi[t] * rng.exponential()
    return x


def test_exponential_acd_is_garch_on_the_square_roots(durations):
    arch = pytest.importorskip("arch")
    fit = acd(_ds(durations), "y")
    omega, a, b = fit.values
    psi = [durations.mean()]
    for t in range(1, len(durations)):
        psi.append(omega + a * durations[t - 1] + b * psi[-1])
    psi = np.array(psi)
    e = np.sqrt(durations)  # Gaussian GARCH quasi-likelihood of sqrt(x) with variance psi, less its constant
    gauss = np.sum(sps.norm.logpdf(e, 0, np.sqrt(psi))) + len(e) * 0.5 * np.log(2 * np.pi)
    assert fit.loglike == pytest.approx(2 * gauss, rel=1e-10)  # twice the Gaussian quasi-likelihood: the same maximiser
    ref = arch.arch_model(e, mean="Zero", vol="GARCH", p=1, q=1).fit(disp="off")
    assert fit.values == pytest.approx(ref.params.to_numpy(), abs=0.01)  # the start of the recursion differs
    for v, true, s in zip(fit.values, (0.1, 0.15, 0.75), fit.se):
        assert v == pytest.approx(true, abs=3 * s)


def test_weibull_acd(durations):
    fit = acd(_ds(durations), "y", distribution="weibull")
    omega, a, b, k = fit.values
    assert k == pytest.approx(1.0, abs=3 * fit.se[3])
    psi = [durations.mean()]
    for t in range(1, len(durations)):
        psi.append(omega + a * durations[t - 1] + b * psi[-1])
    scale = np.array(psi) / special.gamma(1 + 1 / k)  # unit-mean Weibull errors
    assert fit.loglike == pytest.approx(np.sum(sps.weibull_min.logpdf(durations, k, scale=scale)), rel=1e-10)
    with pytest.raises(SpecificationError, match="positive"):
        acd(_ds(durations - 1), "y")


def test_dynamic_probit_is_a_probit_with_a_lag():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(3)
    T = 600
    x = rng.normal(size=T)
    d = np.zeros(T)
    for t in range(1, T):
        d[t] = float(-0.3 + 1.0 * d[t - 1] + 0.8 * x[t] + rng.normal() > 0)
    data = Dataset.from_frame(pd.DataFrame({"d": d, "x": x}, index=pd.period_range("1900-01", periods=T, freq="M")))
    fit = Equation("d ~ d[-1] + x", family=Probit()).fit(data, method=ML())
    ref = sm.Probit(d[1:], sm.add_constant(np.column_stack([d[:-1], x[1:]]))).fit(disp=0)
    assert fit.params.values == pytest.approx(ref.params, rel=1e-6)
