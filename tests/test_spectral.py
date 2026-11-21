"""Spectra and long memory, checked independently: the periodogram against scipy.signal.periodogram (rescaled to the
two-sided density in radians) and integrating to the variance; a smoothed AR(1) spectrum near the true density; the
smoothing as a Daniell average by hand; coherence near 1 for a series and a noisy multiple of it, small for
independent series, and the gain at the multiple; GPH's slope by hand; local Whittle's objective minimised; both, and
ARFIMA, recovering d from a simulated fractional process, ARFIMA(1, d, 0) on an AR(1) finding d near 0."""

import importlib

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, arfima, cross_spectrum, gph, local_whittle, periodogram, spectrum

sp = importlib.import_module("pymetrix.spectral")


def _ds(**cols):
    n = len(next(iter(cols.values())))
    return Dataset.from_frame(pd.DataFrame(cols, index=pd.period_range("1500Q1", periods=n, freq="Q")))


@pytest.fixture(scope="module")
def ar1():
    x = np.random.default_rng(0).normal(size=2000)
    for t in range(1, 2000):
        x[t] += 0.6 * x[t - 1]
    return x


def test_the_periodogram_and_spectra(ar1):
    from scipy.signal import periodogram as scipy_periodogram

    p = periodogram(_ds(y=ar1), "y")
    _, P = scipy_periodogram(ar1, detrend="constant", scaling="density")
    assert p.density[:-1] * 4 * np.pi == pytest.approx(P[1:len(p.density)], rel=1e-10)  # scipy: one-sided, per cycle
    assert 2 * p.density.sum() * 2 * np.pi / len(ar1) == pytest.approx(ar1.var(), rel=1e-3)
    s = spectrum(_ds(y=ar1), "y", bandwidth=10)
    true = 1 / (2 * np.pi * np.abs(1 - 0.6 * np.exp(-1j * s.frequency)) ** 2)
    assert np.median(s.density / true) == pytest.approx(1, abs=0.05)
    j = 100
    assert s.density[j] == pytest.approx(p.density[j - 10:j + 11].mean(), rel=1e-12)  # the Daniell average


def test_coherence_and_gain(ar1):
    rng = np.random.default_rng(1)
    cs = cross_spectrum(_ds(y=2 * ar1 + rng.normal(size=2000) * 0.1, x=ar1), "y", "x")
    assert cs.coherence.mean() > 0.99 and cs.gain.mean() == pytest.approx(2, abs=0.02)
    assert cross_spectrum(_ds(y=rng.normal(size=2000), x=ar1), "y", "x").coherence.mean() < 0.15


@pytest.fixture(scope="module")
def fractional():
    e = np.random.default_rng(2).normal(size=5000)
    return sp.fractional_difference(e, -0.3)[3000:]  # (1 - L)^-0.3 e: d = 0.3


def test_gph_and_local_whittle(fractional):
    data = _ds(y=fractional)
    res = gph(data, "y")
    T, m = len(fractional), int(np.floor(len(fractional) ** 0.5))
    lam, z = sp._dft(fractional)
    I = np.abs(z[:m]) ** 2 / (2 * np.pi * T)
    slope = np.polyfit(-2 * np.log(2 * np.sin(lam[:m] / 2)), np.log(I), 1)[0]
    assert res.statistic == pytest.approx(slope, rel=1e-10) and res.statistic == pytest.approx(0.3, abs=3 * res.details["std err"])
    lw = local_whittle(data, "y")
    m2 = lw.details["frequencies"]
    I2 = np.abs(z[:m2]) ** 2 / (2 * np.pi * T)
    R = lambda d: np.log(np.mean(lam[:m2] ** (2 * d) * I2)) - 2 * d * np.log(lam[:m2]).mean()  # noqa: E731
    assert R(lw.statistic) <= min(R(lw.statistic - 1e-4), R(lw.statistic + 1e-4))
    assert lw.statistic == pytest.approx(0.3, abs=3 * lw.details["std err"])


def test_arfima(fractional, ar1):
    fit = arfima(_ds(y=fractional), "y")
    assert fit.params[0] == pytest.approx(0.3, abs=3 * fit.se[0])
    e = sp._arfima_resid(fit.params, fractional - fractional.mean(), 0, 0)
    assert fit.sigma2 == pytest.approx(e @ e / len(e), rel=1e-12)
    short = arfima(_ds(y=ar1), "y", p=1)
    assert short.params[0] == pytest.approx(0, abs=3 * short.se[0]) and short.params[1] == pytest.approx(0.6, abs=0.1)
