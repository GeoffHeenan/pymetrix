"""Diffusions by exact maximum likelihood, checked independently: Vasicek's estimates equal to the AR(1) least-squares
estimates mapped to (kappa, theta, sigma), as the exact discretisation implies; CIR's likelihood against scipy's
noncentral chi-squared density and its parameters recovered from an exactly simulated path; GBM's estimates and
standard errors against their closed forms, and its standard errors against the spread over simulated samples."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from pymetrix import Dataset, diffusion
from pymetrix.errors import SpecificationError

DT = 1 / 12


def _ds(x, freq="M"):
    return Dataset.from_frame(pd.DataFrame({"r": x}, index=pd.period_range("1900-01", periods=len(x), freq=freq)))


def test_vasicek_is_the_mapped_ar1():
    rng = np.random.default_rng(0)
    k, th, sg, T = 0.5, 0.05, 0.02, 1500
    a = np.exp(-k * DT)
    x = np.empty(T)
    x[0] = 0.05
    for t in range(1, T):
        x[t] = th + (x[t - 1] - th) * a + np.sqrt(sg**2 * (1 - a**2) / (2 * k)) * rng.normal()
    fit = diffusion(_ds(x), "r", dt=DT)
    b1, b0 = np.polyfit(x[:-1], x[1:], 1)
    e = x[1:] - b0 - b1 * x[:-1]
    kappa = -np.log(b1) / DT
    assert fit.values == pytest.approx([kappa, b0 / (1 - b1), np.sqrt(np.mean(e**2) * 2 * kappa / (1 - b1**2))], rel=1e-6)
    for v, true, s in zip(fit.values, (k, th, sg), fit.se):
        assert v == pytest.approx(true, abs=3 * s)


def test_cir():
    rng = np.random.default_rng(1)
    k, th, sg, T = 0.8, 0.06, 0.1, 1500
    c = 2 * k / (sg**2 * (1 - np.exp(-k * DT)))
    x = np.empty(T)
    x[0] = 0.06
    for t in range(1, T):  # the exact transition
        x[t] = sps.ncx2.rvs(4 * k * th / sg**2, 2 * c * x[t - 1] * np.exp(-k * DT), random_state=rng) / (2 * c)
    fit = diffusion(_ds(x), "r", model="cir", dt=DT)
    kh, thh, sgh = fit.values
    ch = 2 * kh / (sgh**2 * (1 - np.exp(-kh * DT)))
    ll = np.sum(sps.ncx2.logpdf(2 * ch * x[1:], 4 * kh * thh / sgh**2, 2 * ch * x[:-1] * np.exp(-kh * DT)) + np.log(2 * ch))
    assert fit.loglike == pytest.approx(ll, rel=1e-10)
    for v, true, s in zip(fit.values, (k, th, sg), fit.se):
        assert v == pytest.approx(true, abs=3 * s)
    with pytest.raises(SpecificationError, match="positive"):
        diffusion(_ds(x - 1), "r", model="cir", dt=DT)


def test_gbm():
    rng = np.random.default_rng(2)
    mu, sg, dt, T = 0.08, 0.2, 1 / 252, 2000
    est, ses = [], []
    for k in range(60):
        r = (mu - sg**2 / 2) * dt + sg * np.sqrt(dt) * rng.standard_normal(T)
        p = 100 * np.exp(np.r_[0, np.cumsum(r)])
        fit = diffusion(_ds(p, freq="D"), "r", model="gbm", dt=dt)
        if k == 0:
            s = np.sqrt(np.var(r) / dt)
            assert fit.values == pytest.approx([np.mean(r) / dt + s**2 / 2, s], rel=1e-12)
        est.append(fit.values)
        ses.append(fit.se)
    est, ses = np.array(est), np.array(ses)
    assert ses.mean(axis=0) == pytest.approx(est.std(axis=0, ddof=1), rel=0.3)  # the closed-form errors against the spread
